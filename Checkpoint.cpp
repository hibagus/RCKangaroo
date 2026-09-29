// Checkpoint / resume support for RCKangaroo. See Checkpoint.h for the format
// and for what the snapshot deliberately leaves out.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#ifdef _WIN32
    #include <io.h>
    #include <windows.h>
    #define CKPT_SYNC(fp)   _commit(_fileno(fp))
#else
    #include <unistd.h>
    #define CKPT_SYNC(fp)   fsync(fileno(fp))
#endif

#include "Checkpoint.h"
#include "utils.h"
#include "GpuKang.h"

// Owned by RCKangaroo.cpp.
extern TFastBase       db;
extern RCGpuKang*      GpuKangs[MAX_GPU_CNT];
extern int             GpuCnt;
extern volatile u64    PntTotalOps;
extern volatile bool   gSolved;
extern EcPoint         gPntToSolve;
extern EcPoint         gPubKey;
extern EcInt           gStart;
extern u32             gRange;
extern u32             gDP;
void CheckNewPoints();

static char gCkptFile[1024];
static int  gCkptIntervalSec;

// A GPU is given this long to reach its pause point. One KernelA/B/C batch is
// STEP_CNT jumps, on the order of 100 ms, so anything close to this means the
// device is wedged and we should save what we can rather than hang the solver.
#define CKPT_PAUSE_TIMEOUT_MS   60000

// Big stdio buffer: the database section writes 16.7M two-byte list counts
// before it writes any records, which is unbearable at the default 4 KB.
#define CKPT_IO_BUF_SIZE        (8 * 1024 * 1024)

void Ckpt_Configure(const char* fn, int interval_min)
{
    if (fn)
    {
        strncpy(gCkptFile, fn, sizeof(gCkptFile) - 1);
        gCkptFile[sizeof(gCkptFile) - 1] = 0;
    }
    if (interval_min > 0)
        gCkptIntervalSec = interval_min * 60;
}

bool        Ckpt_IsEnabled()    { return gCkptFile[0] != 0; }
const char* Ckpt_FileName()     { return gCkptFile; }
int         Ckpt_IntervalSec()  { return gCkptIntervalSec; }
bool        Ckpt_FileExists()   { return gCkptFile[0] && IsFileExist(gCkptFile); }
void        Ckpt_Disable()      { gCkptFile[0] = 0; }

static void CkptTmpName(char* out, int out_size)
{
    snprintf(out, out_size, "%s.tmp", gCkptFile);
}

// rename() refuses to clobber an existing file on Windows.
static bool CkptReplaceFile(const char* from, const char* to)
{
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(from, to) == 0;
#endif
}

static void CkptFillHeader(TCkptHeader* h, u64 elapsed_ms)
{
    memset(h, 0, sizeof(*h));
    memcpy(h->Magic, CKPT_MAGIC, sizeof(h->Magic));
    h->Version   = CKPT_VERSION;
    h->Range     = gRange;
    h->DP        = gDP;
    h->GpuCnt    = (u32)GpuCnt;
    h->BlockSize = BLOCK_SIZE;
    h->GroupCnt  = PNT_GROUP_CNT;
    h->JmpCnt    = JMP_CNT;
    h->StepCnt   = STEP_CNT;
    h->MdLen     = MD_LEN;
    h->TotalOps  = PntTotalOps;
    h->ElapsedMs = elapsed_ms;
    gPntToSolve.SaveToBuffer64(h->PntToSolve);
    gPubKey.SaveToBuffer64(h->PubKey);
    memcpy(h->Start, gStart.data, sizeof(h->Start));
}

// Returns an empty string if the checkpoint describes the same search we are
// running, otherwise a short reason suitable for an error message.
static const char* CkptMismatchReason(const TCkptHeader* h, EcPoint& PntToSolve, int Range, int DP)
{
    if (memcmp(h->Magic, CKPT_MAGIC, sizeof(h->Magic)) != 0)
        return "not an RCKangaroo checkpoint";
    if (h->Version != CKPT_VERSION)
        return "written by a different checkpoint version";
    if ((int)h->Range != Range)
        return "saved with a different -range";
    if ((int)h->DP != DP)
        return "saved with a different -dp";

    u8 pnt[64];
    PntToSolve.SaveToBuffer64(pnt);
    if (memcmp(h->PntToSolve, pnt, sizeof(pnt)) != 0)
        return "saved for a different public key or -start offset";
    return "";
}

bool Ckpt_Load(EcPoint& PntToSolve, int Range, int DP, u64* pTotalOps, u64* pElapsedMs)
{
    if (!Ckpt_IsEnabled())
        return false;

    FILE* fp = fopen(gCkptFile, "rb");
    if (!fp)
    {
        printf("checkpoint: cannot open %s\r\n", gCkptFile);
        return false;
    }
    setvbuf(fp, NULL, _IOFBF, CKPT_IO_BUF_SIZE);

    TCkptHeader h;
    if (fread(&h, 1, sizeof(h), fp) != sizeof(h))
    {
        printf("checkpoint: %s is truncated\r\n", gCkptFile);
        fclose(fp);
        return false;
    }

    const char* bad = CkptMismatchReason(&h, PntToSolve, Range, DP);
    if (bad[0])
    {
        printf("checkpoint: %s was %s - refusing to resume\r\n", gCkptFile, bad);
        fclose(fp);
        return false;
    }

    // A layout change does not invalidate the database, only the kangaroos.
    bool layout_ok = (h.BlockSize == BLOCK_SIZE) && (h.GroupCnt == PNT_GROUP_CNT) &&
                     (h.JmpCnt == JMP_CNT) && (h.StepCnt == STEP_CNT) && (h.MdLen == MD_LEN);
    if (!layout_ok)
        printf("checkpoint: built with different kernel parameters, kangaroos will be "
               "regenerated (the DP database is still used)\r\n");

    std::vector<TCkptGpuDesc> descs(h.GpuCnt);
    if (h.GpuCnt && fread(&descs[0], sizeof(TCkptGpuDesc), h.GpuCnt, fp) != h.GpuCnt)
    {
        printf("checkpoint: %s is truncated in the GPU table\r\n", gCkptFile);
        fclose(fp);
        return false;
    }

    printf("checkpoint: loading DP database (%llu records)...\r\n", (unsigned long long)h.DbRecCnt);
    u64 t0 = GetTickCount64();
    if (CKPT_FSEEK64(fp, (i64)h.DbOffset, SEEK_SET) != 0 || !db.LoadFromStream(fp))
    {
        printf("checkpoint: failed to read the DP database from %s\r\n", gCkptFile);
        db.Clear();
        fclose(fp);
        return false;
    }
    fclose(fp);
    printf("checkpoint: %llu DPs loaded in %llu s\r\n",
           (unsigned long long)db.GetBlockCnt(), (unsigned long long)((GetTickCount64() - t0) / 1000));

    // Hand each GPU the offset of its own block so it can read it in its own
    // thread during Start(), instead of holding every GPU's kangaroos in host
    // memory at once.
    int restored = 0;
    for (int i = 0; i < GpuCnt; i++)
    {
        if (!layout_ok || i >= (int)h.GpuCnt)
            continue;
        const TCkptGpuDesc& d = descs[i];
        if (!(d.DescFlags & CKPT_GPU_HAS_STATE))
        {
            printf("checkpoint: GPU %d had no saved state, it will start from random kangaroos\r\n",
                   GpuKangs[i]->CudaIndex);
            continue;
        }
        if ((int)d.KangCnt != GpuKangs[i]->KangCnt)
        {
            printf("checkpoint: GPU %d had %u kangaroos, now %d - regenerating this GPU\r\n",
                   GpuKangs[i]->CudaIndex, d.KangCnt, GpuKangs[i]->KangCnt);
            continue;
        }
        GpuKangs[i]->SetResumeSource(gCkptFile, d.Offset);
        restored++;
    }

    *pTotalOps  = h.TotalOps;
    *pElapsedMs = h.ElapsedMs;

    u64 sec = h.ElapsedMs / 1000;
    printf("checkpoint: resuming after 2^%.3f ops and %llud:%02dh:%02dm of work, "
           "kangaroo state restored on %d of %d GPUs\r\n",
           h.TotalOps ? log2((double)h.TotalOps) : 0.0,
           (unsigned long long)(sec / (3600 * 24)),
           (int)((sec % (3600 * 24)) / 3600), (int)((sec % 3600) / 60),
           restored, GpuCnt);
    return true;
}

// Asks every worker to stop at its next batch boundary. Returns the number that
// are either parked or no longer running; anything short of GpuCnt means we
// timed out on somebody.
static int CkptQuiesceGpus()
{
    for (int i = 0; i < GpuCnt; i++)
        GpuKangs[i]->RequestPause();

    u64 t0 = GetTickCount64();
    for (;;)
    {
        int ready = 0;
        for (int i = 0; i < GpuCnt; i++)
            if (GpuKangs[i]->IsQuiesced())
                ready++;
        if (ready == GpuCnt)
            return ready;
        if (GetTickCount64() - t0 > CKPT_PAUSE_TIMEOUT_MS)
        {
            printf("checkpoint: %d of %d GPUs did not reach a batch boundary in %d s\r\n",
                   ready, GpuCnt, CKPT_PAUSE_TIMEOUT_MS / 1000);
            return ready;
        }
        Sleep(5);
    }
}

static void CkptReleaseGpus()
{
    for (int i = 0; i < GpuCnt; i++)
        GpuKangs[i]->ReleasePause();
}

// Writes the whole file to fp. The GPU table and the header's DbOffset are
// backpatched at the end, because a GPU that failed to quiesce shifts every
// following offset.
static bool CkptWriteAll(FILE* fp, u64 elapsed_ms)
{
    TCkptHeader h;
    CkptFillHeader(&h, elapsed_ms);

    std::vector<TCkptGpuDesc> descs(GpuCnt);
    memset(&descs[0], 0, GpuCnt * sizeof(TCkptGpuDesc));

    if (fwrite(&h, 1, sizeof(h), fp) != sizeof(h))
        return false;
    i64 table_ofs = CKPT_FTELL64(fp);
    if (fwrite(&descs[0], sizeof(TCkptGpuDesc), GpuCnt, fp) != (size_t)GpuCnt)
        return false;

    u64 kang_total = 0;
    for (int i = 0; i < GpuCnt; i++)
    {
        descs[i].CudaIndex = GpuKangs[i]->CudaIndex;
        descs[i].KangCnt   = GpuKangs[i]->KangCnt;
        if (!GpuKangs[i]->IsQuiesced() || !GpuKangs[i]->IsStateReadable())
            continue;   // left with CKPT_GPU_HAS_STATE clear

        i64 ofs = CKPT_FTELL64(fp);
        if (!GpuKangs[i]->SaveKangsToStream(fp))
        {
            printf("checkpoint: GPU %d state could not be read, saving without it\r\n",
                   GpuKangs[i]->CudaIndex);
            if (CKPT_FSEEK64(fp, ofs, SEEK_SET) != 0)
                return false;   // cannot rewind, the file would be corrupt
            continue;
        }
        descs[i].DescFlags = CKPT_GPU_HAS_STATE;
        descs[i].Offset    = (u64)ofs;
        descs[i].Bytes     = (u64)GpuKangs[i]->KangCnt * 96;
        kang_total        += GpuKangs[i]->KangCnt;
    }

    h.KangTotal = kang_total;
    h.DbOffset  = (u64)CKPT_FTELL64(fp);
    h.DbRecCnt  = db.GetBlockCnt();
    if (!db.SaveToStream(fp))
        return false;

    if (CKPT_FSEEK64(fp, 0, SEEK_SET) != 0)
        return false;
    if (fwrite(&h, 1, sizeof(h), fp) != sizeof(h))
        return false;
    if (CKPT_FSEEK64(fp, table_ofs, SEEK_SET) != 0)
        return false;
    if (fwrite(&descs[0], sizeof(TCkptGpuDesc), GpuCnt, fp) != (size_t)GpuCnt)
        return false;

    return fflush(fp) == 0;
}

bool Ckpt_Save(u64 elapsed_ms)
{
    if (!Ckpt_IsEnabled())
        return false;

    u64 t0 = GetTickCount64();
    printf("checkpoint: saving %s ...\r\n", gCkptFile);
    CkptQuiesceGpus();

    // The workers are parked, so this drains every DP they produced without
    // racing them, and nothing new arrives while we write.
    CheckNewPoints();

    // That flush can be the one that closes the collision. Writing gigabytes
    // for a search that is already over only delays printing the key.
    if (gSolved)
    {
        printf("checkpoint: key found while flushing, nothing to save\r\n");
        CkptReleaseGpus();
        return false;
    }

    char tmp[1100];
    CkptTmpName(tmp, sizeof(tmp));

    bool ok = false;
    FILE* fp = fopen(tmp, "wb");
    if (!fp)
        printf("checkpoint: cannot create %s\r\n", tmp);
    else
    {
        setvbuf(fp, NULL, _IOFBF, CKPT_IO_BUF_SIZE);
        ok = CkptWriteAll(fp, elapsed_ms);
        if (ok)
            CKPT_SYNC(fp);      // survive a machine crash, not just a process exit
        if (fclose(fp) != 0)
            ok = false;
        if (ok)
        {
            ok = CkptReplaceFile(tmp, gCkptFile);
            if (!ok)
                printf("checkpoint: cannot replace %s\r\n", gCkptFile);
        }
        else
            printf("checkpoint: write to %s failed (disk full?)\r\n", tmp);
        if (!ok)
            remove(tmp);
    }

    CkptReleaseGpus();

    if (ok)
        printf("checkpoint: saved %s (%llu DPs) in %llu ms\r\n", gCkptFile,
               (unsigned long long)db.GetBlockCnt(),
               (unsigned long long)(GetTickCount64() - t0));
    return ok;
}
