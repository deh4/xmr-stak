
#include "b_blake256.h"

#pragma pack(push, 1)
typedef struct  
{ 
  u32 h[8], s[4], t[2];
  int buflen, nullt;
  u8  buf[64];
} state ALIGN(64);
#pragma pack(pop)


static const u8 sigma[][16] = 
{
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13 ,0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13 ,0 }};

static const u32 cst[16] = 
{
  0x243F6A88,0x85A308D3,0x13198A2E,0x03707344,
  0xA4093822,0x299F31D0,0x082EFA98,0xEC4E6C89,
  0x452821E6,0x38D01377,0xBE5466CF,0x34E90C6C,
  0xC0AC29B7,0xC97C50DD,0x3F84D5B5,0xB5470917
};

static const u8 padding[] =
{
    0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

static int blake256_compress( state * state, const u8 * datablock ) 
{
  __m128i row1,row2,row3,row4;
  __m128i buf1,buf2,buf3,buf4;
  __m128i s0, s1, s2, s3;
  __m128i t0, t1, t2, t3; 
#ifdef HAVE_SSSE3
  const __m128i u8to32 = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
  const __m128i r8 = _mm_set_epi8(12,15,14,13,8,11,10,9,4,7,6,5,0,3,2,1);
  const __m128i r16 = _mm_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2);
#endif

#ifdef CACHE_MESSAGE
  __m128i cache[4*4] ALIGN(64);
#endif

  static const int sig[][16] = 
  {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }  
  };
  
  static const u32 z[16] = 
  {
    0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
    0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
    0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
    0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917 
  };
  
  const __m128i m0 = _mm_shuffle_epi8(LOADU(datablock +  00), u8to32);
  const __m128i m1 = _mm_shuffle_epi8(LOADU(datablock +  16), u8to32);
  const __m128i m2 = _mm_shuffle_epi8(LOADU(datablock +  32), u8to32);
  const __m128i m3 = _mm_shuffle_epi8(LOADU(datablock +  48), u8to32);

#ifdef NO_BRANCHING
  SET_INITIAL_STATE(row1,row2,row3,row4);
#else
  row1 = LOAD(&state->h[0]); 
  row2 = LOAD(&state->h[4]);
  row3 = _mm_set_epi32(0x03707344, 0x13198A2E, 0x85A308D3, 0x243F6A88); 
  row4 = _mm_set_epi32(0xEC4E6C89, 0x082EFA98, 0x299F31D0, 0xA4093822);
  if (LIKELY(!state->nullt))
  {
    t0 = _mm_cvtsi32_si128(state->t[0]);
    t1 = _mm_cvtsi32_si128(state->t[1]);
    s0 = TOI(_mm_shuffle_ps(TOF(t0), TOF(t1), _MM_SHUFFLE(0,0,0,0)));
    row4 = _mm_xor_si128(row4, s0);
  }
#endif
      
  ROUND( 0);
  ROUND( 1);
  ROUND( 2);
  ROUND( 3);
  ROUND( 4);
  ROUND( 5);
  ROUND( 6);
  ROUND( 7);
  ROUND( 8);
  ROUND( 9);
  ROUND(10);
  ROUND(11);
  ROUND(12);
  ROUND(13);

  t0 = LOAD(state->h);
  t0 = _mm_xor_si128(t0, _mm_xor_si128(row1,row3));
  STORE(state->h, t0);
  
  t1 = LOAD(&(state->h[4]));
  t1 = _mm_xor_si128(t1, _mm_xor_si128(row2,row4));
  STORE(&(state->h[4]), t1);

#ifdef __AVX__
  _mm256_zeroupper();
#endif

  return 0;
}


static void blake256_init( state *S ) 
{
  memset(S, 0, sizeof(state));
  _mm_store_si128((__m128i*)(&S->h[0]), _mm_set_epi32(0xA54FF53A,0x3C6EF372,0xBB67AE85,0x6A09E667));
  _mm_store_si128((__m128i*)(&S->h[4]), _mm_set_epi32(0x5BE0CD19,0x1F83D9AB,0x9B05688C,0x510E527F));
}


static void blake256_update( state *S, const u8 *data, u64 datalen ) 
{
  int left=S->buflen >> 3; 
  int fill=64 - left;
   
  if( left && ( ((datalen >> 3) & 0x3F) >= fill ) ) {
    memcpy( (void*) (S->buf + left), (void*) data, fill );
    S->t[0] += 512;
    if (S->t[0] == 0) S->t[1]++;      
    blake256_compress( S, S->buf );
    data += fill;
    datalen  -= (fill << 3);       
    left = 0;
  }

  while( datalen >= 512 ) {
    S->t[0] += 512;
    if (S->t[0] == 0) S->t[1]++;
    blake256_compress( S, data );
    data += 64;
    datalen  -= 512;
  }
  
  if( datalen > 0 ) {
    memcpy( (void*) (S->buf + left), (void*) data, datalen>>3 );
    S->buflen = (left<<3) + datalen;
  }
  else S->buflen=0;
}


static void blake256_final( state *S, u8 *digest ) 
{
  u8 msglen[8], zo=0x01, oo=0x81;
  u32 lo=S->t[0] + S->buflen, hi=S->t[1];
  if ( lo < S->buflen ) hi++;
  U32TO8(  msglen + 0, hi );
  U32TO8(  msglen + 4, lo );

  if ( S->buflen == 440 ) { /* one padding byte */
    S->t[0] -= 8;
    blake256_update( S, &oo, 8 );
  }
  else {
    if ( S->buflen < 440 ) { /* enough space to fill the block  */
      if ( !S->buflen ) S->nullt=1;
      S->t[0] -= 440 - S->buflen;
      blake256_update( S, padding, 440 - S->buflen );
    }
    else { /* need 2 compressions */
      S->t[0] -= 512 - S->buflen; 
      blake256_update( S, padding, 512 - S->buflen );
      S->t[0] -= 440;
      blake256_update( S, padding+1, 440 );
      S->nullt = 1;
    }
    blake256_update( S, &zo, 8 );
    S->t[0] -= 8;
  }
  S->t[0] -= 64;
  blake256_update( S, msglen, 64 );    
  
  do
  {
#ifdef HAVE_SSSE3
  const __m128i u8to32 = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
#endif
    __m128i w0 = LOAD(&S->h[0]); BSWAP32(w0);
    __m128i w1 = LOAD(&S->h[4]); BSWAP32(w1);
    STOREU((digest + 00), w0);
    STOREU((digest + 16), w1);
  } while(0);
}


void blake256_hash( unsigned char *out, const unsigned char *in, unsigned long long inlen ) {

  state S;  
  blake256_init( &S );
  blake256_update( &S, in, inlen*8 );
  blake256_final( &S, out );

}


