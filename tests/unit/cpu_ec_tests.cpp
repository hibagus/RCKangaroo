#include "Ec.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using boost::multiprecision::cpp_int;

namespace {

constexpr std::string_view kPrimeHex =
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
constexpr std::string_view kGeneratorXHex =
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
constexpr std::string_view kGeneratorYHex =
    "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

int failures = 0;

cpp_int ParseHex(std::string_view text)
{
    cpp_int result = 0;
    for (const char digit : text) {
        result <<= 4;
        if (digit >= '0' && digit <= '9') {
            result += digit - '0';
        } else if (digit >= 'A' && digit <= 'F') {
            result += digit - 'A' + 10;
        } else if (digit >= 'a' && digit <= 'f') {
            result += digit - 'a' + 10;
        }
    }
    return result;
}

const cpp_int kPrime = ParseHex(kPrimeHex);
const cpp_int kGeneratorX = ParseHex(kGeneratorXHex);
const cpp_int kGeneratorY = ParseHex(kGeneratorYHex);
const cpp_int kLimbMask = (cpp_int(1) << 64) - 1;

cpp_int Normalize(cpp_int value)
{
    value %= kPrime;
    if (value < 0) {
        value += kPrime;
    }
    return value;
}

EcInt ToEcInt(cpp_int value)
{
    EcInt result;
    value = Normalize(value);
    for (int limb = 0; limb < 4; ++limb) {
        result.data[limb] = static_cast<u64>((value & kLimbMask).convert_to<std::uint64_t>());
        value >>= 64;
    }
    result.data[4] = 0;
    return result;
}

cpp_int ToCppInt(const EcInt& value)
{
    cpp_int result = 0;
    for (int limb = 3; limb >= 0; --limb) {
        result <<= 64;
        result += value.data[limb];
    }
    return result;
}

std::string ToHex(const cpp_int& value)
{
    std::stringstream stream;
    stream << std::hex << value;
    return stream.str();
}

void Expect(bool condition, std::string_view label)
{
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}

void ExpectFieldEqual(std::string_view label, const EcInt& actual, cpp_int expected)
{
    expected = Normalize(expected);
    const cpp_int actual_value = ToCppInt(actual);
    if (actual.data[4] != 0 || actual_value != expected) {
        ++failures;
        std::cerr << "FAIL: " << label << "\n  expected: 0x" << ToHex(expected)
                  << "\n  actual:   0x" << ToHex(actual_value)
                  << "\n  high limb: " << actual.data[4] << '\n';
    }
}

cpp_int RandomFieldElement(std::mt19937_64& generator)
{
    cpp_int value = 0;
    for (int limb = 0; limb < 4; ++limb) {
        value <<= 64;
        value += generator();
    }
    return value % kPrime;
}

struct ReferencePoint {
    cpp_int x;
    cpp_int y;
    bool infinity = false;
};

cpp_int Inverse(const cpp_int& value)
{
    return boost::multiprecision::powm(Normalize(value), kPrime - 2, kPrime);
}

ReferencePoint AddReferencePoints(const ReferencePoint& left, const ReferencePoint& right)
{
    if (left.infinity) {
        return right;
    }
    if (right.infinity) {
        return left;
    }

    cpp_int slope;
    if (left.x == right.x) {
        if (Normalize(left.y + right.y) == 0) {
            return {{}, {}, true};
        }
        slope = Normalize(3 * left.x * left.x * Inverse(2 * left.y));
    } else {
        slope = Normalize((right.y - left.y) * Inverse(right.x - left.x));
    }

    const cpp_int x = Normalize(slope * slope - left.x - right.x);
    const cpp_int y = Normalize(slope * (left.x - x) - left.y);
    return {x, y, false};
}

ReferencePoint MultiplyReferencePoint(std::uint64_t scalar)
{
    ReferencePoint result{{}, {}, true};
    ReferencePoint addend{kGeneratorX, kGeneratorY, false};
    while (scalar != 0) {
        if ((scalar & 1U) != 0) {
            result = AddReferencePoints(result, addend);
        }
        addend = AddReferencePoints(addend, addend);
        scalar >>= 1;
    }
    return result;
}

void ExpectPointEqual(std::string_view label, const EcPoint& actual,
                      const ReferencePoint& expected)
{
    if (expected.infinity) {
        Expect(false, label);
        return;
    }
    ExpectFieldEqual(std::string(label) + " x", actual.x, expected.x);
    ExpectFieldEqual(std::string(label) + " y", actual.y, expected.y);
}

void TestParsing()
{
    EcInt value;
    Expect(value.SetHexStr("0"), "parse zero");
    ExpectFieldEqual("parsed zero", value, 0);

    const std::string prime_minus_one = ToHex(kPrime - 1);
    Expect(value.SetHexStr(prime_minus_one.c_str()), "parse p - 1");
    ExpectFieldEqual("parsed p - 1", value, kPrime - 1);

    std::array<char, 65> round_trip{};
    value.GetHexStr(round_trip.data());
    Expect(std::string_view(round_trip.data()) ==
               "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E",
           "format p - 1");

    Expect(!value.SetHexStr("xyz"), "reject non-hex integer");
    Expect(!value.SetHexStr(
               "10000000000000000000000000000000000000000000000000000000000000000"),
           "reject integer wider than 256 bits");

    const std::string compressed = "02" + std::string(kGeneratorXHex);
    const std::string uncompressed =
        "04" + std::string(kGeneratorXHex) + std::string(kGeneratorYHex);
    EcPoint point;
    Expect(point.SetHexStr(compressed.c_str()), "parse compressed generator");
    ExpectPointEqual("compressed generator", point,
                     {kGeneratorX, kGeneratorY, false});
    Expect(point.SetHexStr(uncompressed.c_str()), "parse uncompressed generator");
    ExpectPointEqual("uncompressed generator", point,
                     {kGeneratorX, kGeneratorY, false});
    const std::string invalid_prefix = "05" + std::string(kGeneratorXHex);
    Expect(!point.SetHexStr(invalid_prefix.c_str()), "reject invalid point prefix");
}

void TestBoundaryFieldArithmetic()
{
    std::vector<cpp_int> values = {
        0,
        1,
        2,
        (cpp_int(1) << 32) - 1,
        (cpp_int(1) << 64) - 1,
        cpp_int(1) << 128,
        kPrime - 2,
        kPrime - 1,
    };

    for (const cpp_int& left_value : values) {
        EcInt negated = ToEcInt(left_value);
        negated.NegModP();
        ExpectFieldEqual("boundary negate", negated, -left_value);

        for (const cpp_int& right_value : values) {
            EcInt right = ToEcInt(right_value);

            EcInt sum = ToEcInt(left_value);
            sum.AddModP(right);
            ExpectFieldEqual("boundary add", sum, left_value + right_value);

            EcInt difference = ToEcInt(left_value);
            difference.SubModP(right);
            ExpectFieldEqual("boundary subtract", difference, left_value - right_value);

            EcInt product = ToEcInt(left_value);
            product.MulModP(right);
            ExpectFieldEqual("boundary multiply", product, left_value * right_value);
        }
    }

    EcInt zero;
    zero.SetZero();
    EcInt one;
    one.Set(1);

    const std::string modulus_hex(kPrimeHex);
    EcInt modulus;
    Expect(modulus.SetHexStr(modulus_hex.c_str()), "parse modulus boundary");
    EcInt reduced_modulus = modulus;
    reduced_modulus.AddModP(zero);
    ExpectFieldEqual("reduce modulus with add", reduced_modulus, 0);
    EcInt multiplied_modulus = modulus;
    multiplied_modulus.MulModP(one);
    ExpectFieldEqual("reduce modulus with multiply", multiplied_modulus, 0);

    const cpp_int maximum_value = (cpp_int(1) << 256) - 1;
    const std::string maximum_hex(64, 'F');
    EcInt maximum;
    Expect(maximum.SetHexStr(maximum_hex.c_str()), "parse maximum 256-bit value");
    EcInt reduced_maximum = maximum;
    reduced_maximum.AddModP(zero);
    ExpectFieldEqual("reduce maximum with add", reduced_maximum, maximum_value);
    EcInt multiplied_maximum = maximum;
    multiplied_maximum.MulModP(one);
    ExpectFieldEqual("reduce maximum with multiply", multiplied_maximum, maximum_value);
}

void TestRandomFieldArithmetic()
{
    std::mt19937_64 generator(0x52434B414E474152ULL);
    for (int iteration = 0; iteration < 1024; ++iteration) {
        const cpp_int left_value = RandomFieldElement(generator);
        const cpp_int right_value = RandomFieldElement(generator);
        EcInt right = ToEcInt(right_value);

        EcInt sum = ToEcInt(left_value);
        sum.AddModP(right);
        ExpectFieldEqual("random add", sum, left_value + right_value);

        EcInt difference = ToEcInt(left_value);
        difference.SubModP(right);
        ExpectFieldEqual("random subtract", difference, left_value - right_value);

        EcInt product = ToEcInt(left_value);
        product.MulModP(right);
        ExpectFieldEqual("random multiply", product, left_value * right_value);

        EcInt square = ToEcInt(left_value);
        EcInt duplicate = square;
        square.MulModP(duplicate);
        ExpectFieldEqual("random square", square, left_value * left_value);
    }

    for (int iteration = 0; iteration < 64; ++iteration) {
        cpp_int value = RandomFieldElement(generator);
        if (value == 0) {
            value = 1;
        }
        EcInt inverse = ToEcInt(value);
        inverse.InvModP();
        ExpectFieldEqual("random inverse", inverse, Inverse(value));

        EcInt original = ToEcInt(value);
        original.MulModP(inverse);
        ExpectFieldEqual("inverse product", original, 1);
    }

    for (int iteration = 0; iteration < 16; ++iteration) {
        const cpp_int value = RandomFieldElement(generator);
        const cpp_int squared_value = Normalize(value * value);
        EcInt root = ToEcInt(squared_value);
        root.SqrtModP();
        EcInt squared_root = root;
        squared_root.MulModP(root);
        ExpectFieldEqual("square root verification", squared_root, squared_value);
    }
}

void TestPointArithmetic()
{
    struct KnownScalarPoint {
        std::uint64_t scalar;
        std::string_view x;
        std::string_view y;
    };
    constexpr std::array<KnownScalarPoint, 2> known_points = {{
        {2,
         "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",
         "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A"},
        {3,
         "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",
         "388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672"},
    }};

    const std::string compressed = "02" + std::string(kGeneratorXHex);
    EcPoint generator;
    Expect(generator.SetHexStr(compressed.c_str()), "load generator for point tests");
    Expect(Ec::IsValidPoint(generator), "generator lies on secp256k1");

    EcPoint doubled = Ec::DoublePoint(generator);
    ExpectPointEqual("generator double", doubled, MultiplyReferencePoint(2));
    Expect(Ec::IsValidPoint(doubled), "doubled generator lies on secp256k1");

    EcPoint tripled = Ec::AddPoints(generator, doubled);
    ExpectPointEqual("generator plus double", tripled, MultiplyReferencePoint(3));
    Expect(Ec::IsValidPoint(tripled), "tripled generator lies on secp256k1");

    for (const KnownScalarPoint& known : known_points) {
        EcInt scalar_value;
        scalar_value.Set(known.scalar);
        EcPoint actual = Ec::MultiplyG(scalar_value);
        ExpectFieldEqual("known scalar multiplication x", actual.x, ParseHex(known.x));
        ExpectFieldEqual("known scalar multiplication y", actual.y, ParseHex(known.y));
    }

    for (const std::uint64_t scalar : {1ULL, 2ULL, 3ULL, 7ULL, 19ULL, 255ULL,
                                       12345ULL}) {
        EcInt scalar_value;
        scalar_value.Set(scalar);
        EcPoint actual = Ec::MultiplyG(scalar_value);
        ExpectPointEqual("scalar multiplication " + std::to_string(scalar), actual,
                         MultiplyReferencePoint(scalar));
        Expect(Ec::IsValidPoint(actual),
               "scalar multiplication result lies on secp256k1");
    }
}

} // namespace

int main()
{
    InitEc();
    TestParsing();
    TestBoundaryFieldArithmetic();
    TestRandomFieldArithmetic();
    TestPointArithmetic();
    DeInitEc();

    if (failures != 0) {
        std::cerr << failures << " CPU ECC regression test(s) failed\n";
        return 1;
    }

    std::cout << "All CPU ECC regression tests passed\n";
    return 0;
}
