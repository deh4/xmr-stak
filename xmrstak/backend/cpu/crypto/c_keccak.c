// keccak.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

#include <stdint.h>
#include <memory.h>
//#include <emmintrin.h>
#if defined(__GNUC__)
#define ALIGN(x) __attribute__ ((aligned(x)))
#include <x86intrin.h> /* gcc specific, also works for icc */
#elif defined(_MSC_VER)
#define ALIGN(x)
#include <intrin.h>
#elif defined(__ARMCC_VERSION)
#define ALIGN(x) __align(x)
#else
#define ALIGN(x)
#endif
#define HASH_DATA_AREA 136
#define KECCAK_ROUNDS 24

#if 0 //Old version
#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif
#else // New version
typedef unsigned char UINT8;
typedef unsigned long long int UINT64;

typedef __m128i V64;
typedef __m128i V128;
typedef union {
	V128 v128;
	UINT64 v64[2];
} V6464;

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

#define ANDnu64(a, b)       _mm_andnot_si128(a, b)
#define LOAD64(a)           _mm_loadl_epi64((const V64 *)&(a))
#define CONST64(a)          _mm_loadl_epi64((const V64 *)&(a))
#define ROL64(a, o)         _mm_or_si128(_mm_slli_epi64(a, o), _mm_srli_epi64(a, 64-(o)))
#define STORE64(a, b)       _mm_storel_epi64((V64 *)&(a), b)
#define XOR64(a, b)         _mm_xor_si128(a, b)
#define XOReq64(a, b)       a = _mm_xor_si128(a, b)

#define ANDnu128(a, b)      _mm_andnot_si128(a, b)
#define LOAD6464(a, b)      _mm_set_epi64((__m64)(a), (__m64)(b))
#define LOAD128(a)          _mm_load_si128((const V128 *)&(a))
#define LOAD128u(a)         _mm_loadu_si128((const V128 *)&(a))
#define ROL64in128(a, o)    _mm_or_si128(_mm_slli_epi64(a, o), _mm_srli_epi64(a, 64-(o)))
#define STORE128(a, b)      _mm_store_si128((V128 *)&(a), b)
#define XOR128(a, b)        _mm_xor_si128(a, b)
#define XOReq128(a, b)      a = _mm_xor_si128(a, b)
#define GET64LO(a, b)       _mm_unpacklo_epi64(a, b)
#define GET64HI(a, b)       _mm_unpackhi_epi64(a, b)
#define COPY64HI2LO(a)      _mm_shuffle_epi32(a, 0xEE)
#define COPY64LO2HI(a)      _mm_shuffle_epi32(a, 0x44)
#define ZERO128()           _mm_setzero_si128()

#define declareABCDE \
    V64 Aba, Abe, Abi, Abo, Abu; \
    V64 Aga, Age, Agi, Ago, Agu; \
    V64 Aka, Ake, Aki, Ako, Aku; \
    V64 Ama, Ame, Ami, Amo, Amu; \
    V64 Asa, Ase, Asi, Aso, Asu; \
    V64 Bba, Bbe, Bbi, Bbo, Bbu; \
    V64 Bga, Bge, Bgi, Bgo, Bgu; \
    V64 Bka, Bke, Bki, Bko, Bku; \
    V64 Bma, Bme, Bmi, Bmo, Bmu; \
    V64 Bsa, Bse, Bsi, Bso, Bsu; \
    V64 Ca, Ce, Ci, Co, Cu; \
    V64 Da, De, Di, Do, Du; \
    V64 Eba, Ebe, Ebi, Ebo, Ebu; \
    V64 Ega, Ege, Egi, Ego, Egu; \
    V64 Eka, Eke, Eki, Eko, Eku; \
    V64 Ema, Eme, Emi, Emo, Emu; \
    V64 Esa, Ese, Esi, Eso, Esu; \

#define prepareTheta \
    Ca = XOR64(Aba, XOR64(Aga, XOR64(Aka, XOR64(Ama, Asa)))); \
    Ce = XOR64(Abe, XOR64(Age, XOR64(Ake, XOR64(Ame, Ase)))); \
    Ci = XOR64(Abi, XOR64(Agi, XOR64(Aki, XOR64(Ami, Asi)))); \
    Co = XOR64(Abo, XOR64(Ago, XOR64(Ako, XOR64(Amo, Aso)))); \
    Cu = XOR64(Abu, XOR64(Agu, XOR64(Aku, XOR64(Amu, Asu)))); \

// --- Code for round, with prepare-theta
// --- 64-bit lanes mapped to 64-bit words
#define thetaRhoPiChiIotaPrepareTheta(i, A, E) \
    Da = XOR64(Cu, ROL64(Ce, 1)); \
    De = XOR64(Ca, ROL64(Ci, 1)); \
    Di = XOR64(Ce, ROL64(Co, 1)); \
    Do = XOR64(Ci, ROL64(Cu, 1)); \
    Du = XOR64(Co, ROL64(Ca, 1)); \
\
    XOReq64(A##ba, Da); \
    Bba = A##ba; \
    XOReq64(A##ge, De); \
    Bbe = ROL64(A##ge, 44); \
    XOReq64(A##ki, Di); \
    Bbi = ROL64(A##ki, 43); \
    E##ba = XOR64(Bba, ANDnu64(Bbe, Bbi)); \
    XOReq64(E##ba, CONST64(KeccakF1600RoundConstants[i])); \
    Ca = E##ba; \
    XOReq64(A##mo, Do); \
    Bbo = ROL64(A##mo, 21); \
    E##be = XOR64(Bbe, ANDnu64(Bbi, Bbo)); \
    Ce = E##be; \
    XOReq64(A##su, Du); \
    Bbu = ROL64(A##su, 14); \
    E##bi = XOR64(Bbi, ANDnu64(Bbo, Bbu)); \
    Ci = E##bi; \
    E##bo = XOR64(Bbo, ANDnu64(Bbu, Bba)); \
    Co = E##bo; \
    E##bu = XOR64(Bbu, ANDnu64(Bba, Bbe)); \
    Cu = E##bu; \
\
    XOReq64(A##bo, Do); \
    Bga = ROL64(A##bo, 28); \
    XOReq64(A##gu, Du); \
    Bge = ROL64(A##gu, 20); \
    XOReq64(A##ka, Da); \
    Bgi = ROL64(A##ka, 3); \
    E##ga = XOR64(Bga, ANDnu64(Bge, Bgi)); \
    XOReq64(Ca, E##ga); \
    XOReq64(A##me, De); \
    Bgo = ROL64(A##me, 45); \
    E##ge = XOR64(Bge, ANDnu64(Bgi, Bgo)); \
    XOReq64(Ce, E##ge); \
    XOReq64(A##si, Di); \
    Bgu = ROL64(A##si, 61); \
    E##gi = XOR64(Bgi, ANDnu64(Bgo, Bgu)); \
    XOReq64(Ci, E##gi); \
    E##go = XOR64(Bgo, ANDnu64(Bgu, Bga)); \
    XOReq64(Co, E##go); \
    E##gu = XOR64(Bgu, ANDnu64(Bga, Bge)); \
    XOReq64(Cu, E##gu); \
\
    XOReq64(A##be, De); \
    Bka = ROL64(A##be, 1); \
    XOReq64(A##gi, Di); \
    Bke = ROL64(A##gi, 6); \
    XOReq64(A##ko, Do); \
    Bki = ROL64(A##ko, 25); \
    E##ka = XOR64(Bka, ANDnu64(Bke, Bki)); \
    XOReq64(Ca, E##ka); \
    XOReq64(A##mu, Du); \
    Bko = ROL64(A##mu, 8); \
    E##ke = XOR64(Bke, ANDnu64(Bki, Bko)); \
    XOReq64(Ce, E##ke); \
    XOReq64(A##sa, Da); \
    Bku = ROL64(A##sa, 18); \
    E##ki = XOR64(Bki, ANDnu64(Bko, Bku)); \
    XOReq64(Ci, E##ki); \
    E##ko = XOR64(Bko, ANDnu64(Bku, Bka)); \
    XOReq64(Co, E##ko); \
    E##ku = XOR64(Bku, ANDnu64(Bka, Bke)); \
    XOReq64(Cu, E##ku); \
\
    XOReq64(A##bu, Du); \
    Bma = ROL64(A##bu, 27); \
    XOReq64(A##ga, Da); \
    Bme = ROL64(A##ga, 36); \
    XOReq64(A##ke, De); \
    Bmi = ROL64(A##ke, 10); \
    E##ma = XOR64(Bma, ANDnu64(Bme, Bmi)); \
    XOReq64(Ca, E##ma); \
    XOReq64(A##mi, Di); \
    Bmo = ROL64(A##mi, 15); \
    E##me = XOR64(Bme, ANDnu64(Bmi, Bmo)); \
    XOReq64(Ce, E##me); \
    XOReq64(A##so, Do); \
    Bmu = ROL64(A##so, 56); \
    E##mi = XOR64(Bmi, ANDnu64(Bmo, Bmu)); \
    XOReq64(Ci, E##mi); \
    E##mo = XOR64(Bmo, ANDnu64(Bmu, Bma)); \
    XOReq64(Co, E##mo); \
    E##mu = XOR64(Bmu, ANDnu64(Bma, Bme)); \
    XOReq64(Cu, E##mu); \
\
    XOReq64(A##bi, Di); \
    Bsa = ROL64(A##bi, 62); \
    XOReq64(A##go, Do); \
    Bse = ROL64(A##go, 55); \
    XOReq64(A##ku, Du); \
    Bsi = ROL64(A##ku, 39); \
    E##sa = XOR64(Bsa, ANDnu64(Bse, Bsi)); \
    XOReq64(Ca, E##sa); \
    XOReq64(A##ma, Da); \
    Bso = ROL64(A##ma, 41); \
    E##se = XOR64(Bse, ANDnu64(Bsi, Bso)); \
    XOReq64(Ce, E##se); \
    XOReq64(A##se, De); \
    Bsu = ROL64(A##se, 2); \
    E##si = XOR64(Bsi, ANDnu64(Bso, Bsu)); \
    XOReq64(Ci, E##si); \
    E##so = XOR64(Bso, ANDnu64(Bsu, Bsa)); \
    XOReq64(Co, E##so); \
    E##su = XOR64(Bsu, ANDnu64(Bsa, Bse)); \
    XOReq64(Cu, E##su); \
\

// --- Code for round
// --- 64-bit lanes mapped to 64-bit words
#define thetaRhoPiChiIota(i, A, E) \
    Da = XOR64(Cu, ROL64(Ce, 1)); \
    De = XOR64(Ca, ROL64(Ci, 1)); \
    Di = XOR64(Ce, ROL64(Co, 1)); \
    Do = XOR64(Ci, ROL64(Cu, 1)); \
    Du = XOR64(Co, ROL64(Ca, 1)); \
\
    XOReq64(A##ba, Da); \
    Bba = A##ba; \
    XOReq64(A##ge, De); \
    Bbe = ROL64(A##ge, 44); \
    XOReq64(A##ki, Di); \
    Bbi = ROL64(A##ki, 43); \
    E##ba = XOR64(Bba, ANDnu64(Bbe, Bbi)); \
    XOReq64(E##ba, CONST64(KeccakF1600RoundConstants[i])); \
    XOReq64(A##mo, Do); \
    Bbo = ROL64(A##mo, 21); \
    E##be = XOR64(Bbe, ANDnu64(Bbi, Bbo)); \
    XOReq64(A##su, Du); \
    Bbu = ROL64(A##su, 14); \
    E##bi = XOR64(Bbi, ANDnu64(Bbo, Bbu)); \
    E##bo = XOR64(Bbo, ANDnu64(Bbu, Bba)); \
    E##bu = XOR64(Bbu, ANDnu64(Bba, Bbe)); \
\
    XOReq64(A##bo, Do); \
    Bga = ROL64(A##bo, 28); \
    XOReq64(A##gu, Du); \
    Bge = ROL64(A##gu, 20); \
    XOReq64(A##ka, Da); \
    Bgi = ROL64(A##ka, 3); \
    E##ga = XOR64(Bga, ANDnu64(Bge, Bgi)); \
    XOReq64(A##me, De); \
    Bgo = ROL64(A##me, 45); \
    E##ge = XOR64(Bge, ANDnu64(Bgi, Bgo)); \
    XOReq64(A##si, Di); \
    Bgu = ROL64(A##si, 61); \
    E##gi = XOR64(Bgi, ANDnu64(Bgo, Bgu)); \
    E##go = XOR64(Bgo, ANDnu64(Bgu, Bga)); \
    E##gu = XOR64(Bgu, ANDnu64(Bga, Bge)); \
\
    XOReq64(A##be, De); \
    Bka = ROL64(A##be, 1); \
    XOReq64(A##gi, Di); \
    Bke = ROL64(A##gi, 6); \
    XOReq64(A##ko, Do); \
    Bki = ROL64(A##ko, 25); \
    E##ka = XOR64(Bka, ANDnu64(Bke, Bki)); \
    XOReq64(A##mu, Du); \
    Bko = ROL64(A##mu, 8); \
    E##ke = XOR64(Bke, ANDnu64(Bki, Bko)); \
    XOReq64(A##sa, Da); \
    Bku = ROL64(A##sa, 18); \
    E##ki = XOR64(Bki, ANDnu64(Bko, Bku)); \
    E##ko = XOR64(Bko, ANDnu64(Bku, Bka)); \
    E##ku = XOR64(Bku, ANDnu64(Bka, Bke)); \
\
    XOReq64(A##bu, Du); \
    Bma = ROL64(A##bu, 27); \
    XOReq64(A##ga, Da); \
    Bme = ROL64(A##ga, 36); \
    XOReq64(A##ke, De); \
    Bmi = ROL64(A##ke, 10); \
    E##ma = XOR64(Bma, ANDnu64(Bme, Bmi)); \
    XOReq64(A##mi, Di); \
    Bmo = ROL64(A##mi, 15); \
    E##me = XOR64(Bme, ANDnu64(Bmi, Bmo)); \
    XOReq64(A##so, Do); \
    Bmu = ROL64(A##so, 56); \
    E##mi = XOR64(Bmi, ANDnu64(Bmo, Bmu)); \
    E##mo = XOR64(Bmo, ANDnu64(Bmu, Bma)); \
    E##mu = XOR64(Bmu, ANDnu64(Bma, Bme)); \
\
    XOReq64(A##bi, Di); \
    Bsa = ROL64(A##bi, 62); \
    XOReq64(A##go, Do); \
    Bse = ROL64(A##go, 55); \
    XOReq64(A##ku, Du); \
    Bsi = ROL64(A##ku, 39); \
    E##sa = XOR64(Bsa, ANDnu64(Bse, Bsi)); \
    XOReq64(A##ma, Da); \
    Bso = ROL64(A##ma, 41); \
    E##se = XOR64(Bse, ANDnu64(Bsi, Bso)); \
    XOReq64(A##se, De); \
    Bsu = ROL64(A##se, 2); \
    E##si = XOR64(Bsi, ANDnu64(Bso, Bsu)); \
    E##so = XOR64(Bso, ANDnu64(Bsu, Bsa)); \
    E##su = XOR64(Bsu, ANDnu64(Bsa, Bse)); \
\

const UINT64 KeccakF1600RoundConstants[24] = {
	0x0000000000000001ULL,
	0x0000000000008082ULL,
	0x800000000000808aULL,
	0x8000000080008000ULL,
	0x000000000000808bULL,
	0x0000000080000001ULL,
	0x8000000080008081ULL,
	0x8000000000008009ULL,
	0x000000000000008aULL,
	0x0000000000000088ULL,
	0x0000000080008009ULL,
	0x000000008000000aULL,
	0x000000008000808bULL,
	0x800000000000008bULL,
	0x8000000000008089ULL,
	0x8000000000008003ULL,
	0x8000000000008002ULL,
	0x8000000000000080ULL,
	0x000000000000800aULL,
	0x800000008000000aULL,
	0x8000000080008081ULL,
	0x8000000000008080ULL,
	0x0000000080000001ULL,
	0x8000000080008008ULL };

#define copyFromStateAndXor576bits(X, state, input) \
    X##ba = XOR64(LOAD64(state[ 0]), LOAD64(input[ 0])); \
    X##be = XOR64(LOAD64(state[ 1]), LOAD64(input[ 1])); \
    X##bi = XOR64(LOAD64(state[ 2]), LOAD64(input[ 2])); \
    X##bo = XOR64(LOAD64(state[ 3]), LOAD64(input[ 3])); \
    X##bu = XOR64(LOAD64(state[ 4]), LOAD64(input[ 4])); \
    X##ga = XOR64(LOAD64(state[ 5]), LOAD64(input[ 5])); \
    X##ge = XOR64(LOAD64(state[ 6]), LOAD64(input[ 6])); \
    X##gi = XOR64(LOAD64(state[ 7]), LOAD64(input[ 7])); \
    X##go = XOR64(LOAD64(state[ 8]), LOAD64(input[ 8])); \
    X##gu = LOAD64(state[ 9]); \
    X##ka = LOAD64(state[10]); \
    X##ke = LOAD64(state[11]); \
    X##ki = LOAD64(state[12]); \
    X##ko = LOAD64(state[13]); \
    X##ku = LOAD64(state[14]); \
    X##ma = LOAD64(state[15]); \
    X##me = LOAD64(state[16]); \
    X##mi = LOAD64(state[17]); \
    X##mo = LOAD64(state[18]); \
    X##mu = LOAD64(state[19]); \
    X##sa = LOAD64(state[20]); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyFromStateAndXor832bits(X, state, input) \
    X##ba = XOR64(LOAD64(state[ 0]), LOAD64(input[ 0])); \
    X##be = XOR64(LOAD64(state[ 1]), LOAD64(input[ 1])); \
    X##bi = XOR64(LOAD64(state[ 2]), LOAD64(input[ 2])); \
    X##bo = XOR64(LOAD64(state[ 3]), LOAD64(input[ 3])); \
    X##bu = XOR64(LOAD64(state[ 4]), LOAD64(input[ 4])); \
    X##ga = XOR64(LOAD64(state[ 5]), LOAD64(input[ 5])); \
    X##ge = XOR64(LOAD64(state[ 6]), LOAD64(input[ 6])); \
    X##gi = XOR64(LOAD64(state[ 7]), LOAD64(input[ 7])); \
    X##go = XOR64(LOAD64(state[ 8]), LOAD64(input[ 8])); \
    X##gu = XOR64(LOAD64(state[ 9]), LOAD64(input[ 9])); \
    X##ka = XOR64(LOAD64(state[10]), LOAD64(input[10])); \
    X##ke = XOR64(LOAD64(state[11]), LOAD64(input[11])); \
    X##ki = XOR64(LOAD64(state[12]), LOAD64(input[12])); \
    X##ko = LOAD64(state[13]); \
    X##ku = LOAD64(state[14]); \
    X##ma = LOAD64(state[15]); \
    X##me = LOAD64(state[16]); \
    X##mi = LOAD64(state[17]); \
    X##mo = LOAD64(state[18]); \
    X##mu = LOAD64(state[19]); \
    X##sa = LOAD64(state[20]); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyFromStateAndXor1024bits(X, state, input) \
    X##ba = XOR64(LOAD64(state[ 0]), LOAD64(input[ 0])); \
    X##be = XOR64(LOAD64(state[ 1]), LOAD64(input[ 1])); \
    X##bi = XOR64(LOAD64(state[ 2]), LOAD64(input[ 2])); \
    X##bo = XOR64(LOAD64(state[ 3]), LOAD64(input[ 3])); \
    X##bu = XOR64(LOAD64(state[ 4]), LOAD64(input[ 4])); \
    X##ga = XOR64(LOAD64(state[ 5]), LOAD64(input[ 5])); \
    X##ge = XOR64(LOAD64(state[ 6]), LOAD64(input[ 6])); \
    X##gi = XOR64(LOAD64(state[ 7]), LOAD64(input[ 7])); \
    X##go = XOR64(LOAD64(state[ 8]), LOAD64(input[ 8])); \
    X##gu = XOR64(LOAD64(state[ 9]), LOAD64(input[ 9])); \
    X##ka = XOR64(LOAD64(state[10]), LOAD64(input[10])); \
    X##ke = XOR64(LOAD64(state[11]), LOAD64(input[11])); \
    X##ki = XOR64(LOAD64(state[12]), LOAD64(input[12])); \
    X##ko = XOR64(LOAD64(state[13]), LOAD64(input[13])); \
    X##ku = XOR64(LOAD64(state[14]), LOAD64(input[14])); \
    X##ma = XOR64(LOAD64(state[15]), LOAD64(input[15])); \
    X##me = LOAD64(state[16]); \
    X##mi = LOAD64(state[17]); \
    X##mo = LOAD64(state[18]); \
    X##mu = LOAD64(state[19]); \
    X##sa = LOAD64(state[20]); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyFromStateAndXor1088bits(X, state, input) \
    X##ba = XOR64(LOAD64(state[ 0]), LOAD64(input[ 0])); \
    X##be = XOR64(LOAD64(state[ 1]), LOAD64(input[ 1])); \
    X##bi = XOR64(LOAD64(state[ 2]), LOAD64(input[ 2])); \
    X##bo = XOR64(LOAD64(state[ 3]), LOAD64(input[ 3])); \
    X##bu = XOR64(LOAD64(state[ 4]), LOAD64(input[ 4])); \
    X##ga = XOR64(LOAD64(state[ 5]), LOAD64(input[ 5])); \
    X##ge = XOR64(LOAD64(state[ 6]), LOAD64(input[ 6])); \
    X##gi = XOR64(LOAD64(state[ 7]), LOAD64(input[ 7])); \
    X##go = XOR64(LOAD64(state[ 8]), LOAD64(input[ 8])); \
    X##gu = XOR64(LOAD64(state[ 9]), LOAD64(input[ 9])); \
    X##ka = XOR64(LOAD64(state[10]), LOAD64(input[10])); \
    X##ke = XOR64(LOAD64(state[11]), LOAD64(input[11])); \
    X##ki = XOR64(LOAD64(state[12]), LOAD64(input[12])); \
    X##ko = XOR64(LOAD64(state[13]), LOAD64(input[13])); \
    X##ku = XOR64(LOAD64(state[14]), LOAD64(input[14])); \
    X##ma = XOR64(LOAD64(state[15]), LOAD64(input[15])); \
    X##me = XOR64(LOAD64(state[16]), LOAD64(input[16])); \
    X##mi = LOAD64(state[17]); \
    X##mo = LOAD64(state[18]); \
    X##mu = LOAD64(state[19]); \
    X##sa = LOAD64(state[20]); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyFromStateAndXor1152bits(X, state, input) \
    X##ba = XOR64(LOAD64(state[ 0]), LOAD64(input[ 0])); \
    X##be = XOR64(LOAD64(state[ 1]), LOAD64(input[ 1])); \
    X##bi = XOR64(LOAD64(state[ 2]), LOAD64(input[ 2])); \
    X##bo = XOR64(LOAD64(state[ 3]), LOAD64(input[ 3])); \
    X##bu = XOR64(LOAD64(state[ 4]), LOAD64(input[ 4])); \
    X##ga = XOR64(LOAD64(state[ 5]), LOAD64(input[ 5])); \
    X##ge = XOR64(LOAD64(state[ 6]), LOAD64(input[ 6])); \
    X##gi = XOR64(LOAD64(state[ 7]), LOAD64(input[ 7])); \
    X##go = XOR64(LOAD64(state[ 8]), LOAD64(input[ 8])); \
    X##gu = XOR64(LOAD64(state[ 9]), LOAD64(input[ 9])); \
    X##ka = XOR64(LOAD64(state[10]), LOAD64(input[10])); \
    X##ke = XOR64(LOAD64(state[11]), LOAD64(input[11])); \
    X##ki = XOR64(LOAD64(state[12]), LOAD64(input[12])); \
    X##ko = XOR64(LOAD64(state[13]), LOAD64(input[13])); \
    X##ku = XOR64(LOAD64(state[14]), LOAD64(input[14])); \
    X##ma = XOR64(LOAD64(state[15]), LOAD64(input[15])); \
    X##me = XOR64(LOAD64(state[16]), LOAD64(input[16])); \
    X##mi = XOR64(LOAD64(state[17]), LOAD64(input[17])); \
    X##mo = LOAD64(state[18]); \
    X##mu = LOAD64(state[19]); \
    X##sa = LOAD64(state[20]); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyFromStateAndXor1344bits(X, state, input) \
    X##ba = XOR64(LOAD64(state[ 0]), LOAD64(input[ 0])); \
    X##be = XOR64(LOAD64(state[ 1]), LOAD64(input[ 1])); \
    X##bi = XOR64(LOAD64(state[ 2]), LOAD64(input[ 2])); \
    X##bo = XOR64(LOAD64(state[ 3]), LOAD64(input[ 3])); \
    X##bu = XOR64(LOAD64(state[ 4]), LOAD64(input[ 4])); \
    X##ga = XOR64(LOAD64(state[ 5]), LOAD64(input[ 5])); \
    X##ge = XOR64(LOAD64(state[ 6]), LOAD64(input[ 6])); \
    X##gi = XOR64(LOAD64(state[ 7]), LOAD64(input[ 7])); \
    X##go = XOR64(LOAD64(state[ 8]), LOAD64(input[ 8])); \
    X##gu = XOR64(LOAD64(state[ 9]), LOAD64(input[ 9])); \
    X##ka = XOR64(LOAD64(state[10]), LOAD64(input[10])); \
    X##ke = XOR64(LOAD64(state[11]), LOAD64(input[11])); \
    X##ki = XOR64(LOAD64(state[12]), LOAD64(input[12])); \
    X##ko = XOR64(LOAD64(state[13]), LOAD64(input[13])); \
    X##ku = XOR64(LOAD64(state[14]), LOAD64(input[14])); \
    X##ma = XOR64(LOAD64(state[15]), LOAD64(input[15])); \
    X##me = XOR64(LOAD64(state[16]), LOAD64(input[16])); \
    X##mi = XOR64(LOAD64(state[17]), LOAD64(input[17])); \
    X##mo = XOR64(LOAD64(state[18]), LOAD64(input[18])); \
    X##mu = XOR64(LOAD64(state[19]), LOAD64(input[19])); \
    X##sa = XOR64(LOAD64(state[20]), LOAD64(input[20])); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyFromState(X, state) \
    X##ba = LOAD64(state[ 0]); \
    X##be = LOAD64(state[ 1]); \
    X##bi = LOAD64(state[ 2]); \
    X##bo = LOAD64(state[ 3]); \
    X##bu = LOAD64(state[ 4]); \
    X##ga = LOAD64(state[ 5]); \
    X##ge = LOAD64(state[ 6]); \
    X##gi = LOAD64(state[ 7]); \
    X##go = LOAD64(state[ 8]); \
    X##gu = LOAD64(state[ 9]); \
    X##ka = LOAD64(state[10]); \
    X##ke = LOAD64(state[11]); \
    X##ki = LOAD64(state[12]); \
    X##ko = LOAD64(state[13]); \
    X##ku = LOAD64(state[14]); \
    X##ma = LOAD64(state[15]); \
    X##me = LOAD64(state[16]); \
    X##mi = LOAD64(state[17]); \
    X##mo = LOAD64(state[18]); \
    X##mu = LOAD64(state[19]); \
    X##sa = LOAD64(state[20]); \
    X##se = LOAD64(state[21]); \
    X##si = LOAD64(state[22]); \
    X##so = LOAD64(state[23]); \
    X##su = LOAD64(state[24]); \

#define copyToState(state, X) \
    STORE64(state[ 0], X##ba); \
    STORE64(state[ 1], X##be); \
    STORE64(state[ 2], X##bi); \
    STORE64(state[ 3], X##bo); \
    STORE64(state[ 4], X##bu); \
    STORE64(state[ 5], X##ga); \
    STORE64(state[ 6], X##ge); \
    STORE64(state[ 7], X##gi); \
    STORE64(state[ 8], X##go); \
    STORE64(state[ 9], X##gu); \
    STORE64(state[10], X##ka); \
    STORE64(state[11], X##ke); \
    STORE64(state[12], X##ki); \
    STORE64(state[13], X##ko); \
    STORE64(state[14], X##ku); \
    STORE64(state[15], X##ma); \
    STORE64(state[16], X##me); \
    STORE64(state[17], X##mi); \
    STORE64(state[18], X##mo); \
    STORE64(state[19], X##mu); \
    STORE64(state[20], X##sa); \
    STORE64(state[21], X##se); \
    STORE64(state[22], X##si); \
    STORE64(state[23], X##so); \
    STORE64(state[24], X##su); \

#define copyStateVariables(X, Y) \
    X##ba = Y##ba; \
    X##be = Y##be; \
    X##bi = Y##bi; \
    X##bo = Y##bo; \
    X##bu = Y##bu; \
    X##ga = Y##ga; \
    X##ge = Y##ge; \
    X##gi = Y##gi; \
    X##go = Y##go; \
    X##gu = Y##gu; \
    X##ka = Y##ka; \
    X##ke = Y##ke; \
    X##ki = Y##ki; \
    X##ko = Y##ko; \
    X##ku = Y##ku; \
    X##ma = Y##ma; \
    X##me = Y##me; \
    X##mi = Y##mi; \
    X##mo = Y##mo; \
    X##mu = Y##mu; \
    X##sa = Y##sa; \
    X##se = Y##se; \
    X##si = Y##si; \
    X##so = Y##so; \
    X##su = Y##su; \

#define rounds \
    prepareTheta \
    thetaRhoPiChiIotaPrepareTheta( 0, A, E) \
    thetaRhoPiChiIotaPrepareTheta( 1, E, A) \
    thetaRhoPiChiIotaPrepareTheta( 2, A, E) \
    thetaRhoPiChiIotaPrepareTheta( 3, E, A) \
    thetaRhoPiChiIotaPrepareTheta( 4, A, E) \
    thetaRhoPiChiIotaPrepareTheta( 5, E, A) \
    thetaRhoPiChiIotaPrepareTheta( 6, A, E) \
    thetaRhoPiChiIotaPrepareTheta( 7, E, A) \
    thetaRhoPiChiIotaPrepareTheta( 8, A, E) \
    thetaRhoPiChiIotaPrepareTheta( 9, E, A) \
    thetaRhoPiChiIotaPrepareTheta(10, A, E) \
    thetaRhoPiChiIotaPrepareTheta(11, E, A) \
    thetaRhoPiChiIotaPrepareTheta(12, A, E) \
    thetaRhoPiChiIotaPrepareTheta(13, E, A) \
    thetaRhoPiChiIotaPrepareTheta(14, A, E) \
    thetaRhoPiChiIotaPrepareTheta(15, E, A) \
    thetaRhoPiChiIotaPrepareTheta(16, A, E) \
    thetaRhoPiChiIotaPrepareTheta(17, E, A) \
    thetaRhoPiChiIotaPrepareTheta(18, A, E) \
    thetaRhoPiChiIotaPrepareTheta(19, E, A) \
    thetaRhoPiChiIotaPrepareTheta(20, A, E) \
    thetaRhoPiChiIotaPrepareTheta(21, E, A) \
    thetaRhoPiChiIotaPrepareTheta(22, A, E) \
    thetaRhoPiChiIota(23, E, A) \
    copyToState(st, A)
#endif

const uint64_t keccakf_rndc[24] = 
{
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// update the state with given number of rounds
void keccakff(uint64_t *st, int iRounds)
{
	declareABCDE
	copyFromState(A, st)
	rounds
}
void keccakf(uint64_t st[25], int iRounds)
{
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < iRounds; ++round) {

        // Theta
		//bc[0] = _mm256_xor_si256(__m256i)st[0], (__m256i)st[5]));
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        for (i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            st[i     ] ^= t;
            st[i +  5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        // Rho Pi
        t = st[1];
        st[ 1] = ROTL64(st[ 6], 44);
        st[ 6] = ROTL64(st[ 9], 20);
        st[ 9] = ROTL64(st[22], 61);
        st[22] = ROTL64(st[14], 39);
        st[14] = ROTL64(st[20], 18);
        st[20] = ROTL64(st[ 2], 62);
        st[ 2] = ROTL64(st[12], 43);
        st[12] = ROTL64(st[13], 25);
        st[13] = ROTL64(st[19],  8);
        st[19] = ROTL64(st[23], 56);
        st[23] = ROTL64(st[15], 41);
        st[15] = ROTL64(st[ 4], 27);
        st[ 4] = ROTL64(st[24], 14);
        st[24] = ROTL64(st[21],  2);
        st[21] = ROTL64(st[ 8], 55);
        st[ 8] = ROTL64(st[16], 45);
        st[16] = ROTL64(st[ 5], 36);
        st[ 5] = ROTL64(st[ 3], 28);
        st[ 3] = ROTL64(st[18], 21);
        st[18] = ROTL64(st[17], 15);
        st[17] = ROTL64(st[11], 10);
        st[11] = ROTL64(st[ 7],  6);
        st[ 7] = ROTL64(st[10],  3);
        st[10] = ROTL64(t, 1);

        //  Chi
        // unrolled loop, where only last iteration is different
        j = 0;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 5;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 10;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 15;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 20;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];
        bc[2] = st[j + 2];
        bc[3] = st[j + 3];
        bc[4] = st[j + 4];

        st[j    ] ^= (~bc[1]) & bc[2];
        st[j + 1] ^= (~bc[2]) & bc[3];
        st[j + 2] ^= (~bc[3]) & bc[4];
        st[j + 3] ^= (~bc[4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];
        
        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}

// compute a keccak hash (md) of given byte length from "in"
typedef uint64_t state_t[25];

void keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen)
{
    state_t st;
    uint8_t temp[144];
    int i, rsiz, rsizw;

    rsiz = sizeof(state_t) == mdlen ? HASH_DATA_AREA : 200 - 2 * mdlen;
    rsizw = rsiz / 8;
    
    memset(st, 0, sizeof(st));

    for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (i = 0; i < rsizw; i++)
            st[i] ^= ((uint64_t *) in)[i];
        //keccakf(st, KECCAK_ROUNDS);
		keccakff(&st, 0);
    }
    
    // last block and padding
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++)
        st[i] ^= ((uint64_t *) temp)[i];

    //keccakf(st, KECCAK_ROUNDS);
	keccakff(&st, 0);
    memcpy(md, st, mdlen);
}

void keccak1600(const uint8_t *in, int inlen, uint8_t *md)
{
    keccak(in, inlen, md, sizeof(state_t));
}
