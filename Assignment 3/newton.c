#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <math.h>
#include <stdint.h>
#include <complex.h>
#include <time.h>
#include <string.h>

#define _XOPEN_SOURCE 700
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


typedef uint8_t TYPE_ATTR;
typedef uint8_t TYPE_CONV;

static TYPE_ATTR **attractors = NULL;
static TYPE_CONV **convergences = NULL;

static int nmb_lines = 0;
static int nthrds = 1;
static int degree = 1;

static mtx_t mtx;
static cnd_t cnd;

typedef struct{
 int val;
 char pad[60];
} int_padded;

typedef struct{
 int tx;
 int ib;
 int istep;
 int sz;
 int_padded *status;
} thrd_info_t;

typedef struct{
 int sz;
 int nthrds;
 int_padded *status;
 int conv_cap;
} thrd_info_writer_t;

// define cplx operations for readability and convenience
typedef struct{
 double r;
 double i;
} cplx;

static inline cplx cmul(cplx a, cplx b){
 cplx c = {a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r};
 return c;
}
static inline cplx cadd(cplx a, cplx b){
 cplx c = {a.r + b.r, a.i + b.i};
 return c;
}
static inline cplx csub(cplx a, cplx b){
 cplx c = { a.r - b.r, a.i - b.i };
 return c;
}
static inline cplx cscale(cplx a, double s){
 cplx c = {a.r * s, a.i * s};
 return c;
}
static inline double cnorm2(cplx a){
 return a.r*a.r + a.i*a.i;
}

// precompute the roots for the polynomial x^d-1 for given d
static cplx *roots = NULL;
static void
precompute_roots(
 int d
){
 roots = malloc(d * sizeof(cplx));
 for ( int kx = 0; kx < d; ++kx ){
  double ang = 2.0 * M_PI * (double)kx / (double)d;
  roots[kx].r = cos(ang);
  roots[kx].i = sin(ang);
 }
}

// perform newton iteration and return root index solution and set conv value to pointer
static int
newton_iter(
 cplx x0,
 int deg,
 int *out_conv
){
 // assignment parameters
 const double tol = 1e-3;
 const double big_lim = 1e10;
 const int max_iters = 128;
 const double tol_sq = tol * tol;

 cplx x = x0;
 int conv = 0;
 while ( 1 ){
  // assignment conditions
  if ( cnorm2(x) < tol_sq ){ // if closer than 1e-3 to origin count as diverge
   *out_conv = conv;
   return deg;
  }
  if ( fabs(x.r) > big_lim || fabs(x.i) > big_lim ){ // if real or imaginary part >1e10 count as diverge
   *out_conv = conv;
   return deg;
  }
  if ( conv >= max_iters ){ // if steps taken >= 128 count as diverge
   *out_conv = conv;
   return deg;
  }

  for ( int kx = 0; kx < deg; ++kx ){ // check if we are within tolerance distance to any root and return root index
   cplx d = csub(x, roots[kx]);
   if ( cnorm2(d) < tol_sq ){ // use squared distance instead of having to calculate square root for efficiency
    *out_conv = conv;
    return kx;
   }
  }
  // we use newton step x_n+1 = (d-1)/d  * x + 1/d * x^-(d-1)
  cplx xpow;
  switch ( deg ){
   case 1:{
    x.r = 1.0;
    x.i = 0.0;
    conv += 1;
    continue;
   }
   case 2:{
    xpow = x;
    break;
   }
   case 3:{
    xpow = cmul(x, x);
    break;
   }
   case 4:{
    cplx x2 = cmul(x, x);
    xpow = cmul(x2, x);
    break;
   }
   case 5:{
    cplx x2 = cmul(x, x);
    xpow = cmul(x2, x2);
    break;
   }
   case 6:{
    cplx x2 = cmul(x, x);
    cplx x3 = cmul(x2, x);
    xpow = cmul(x2, x3);
    break;
   }
   case 7:{
    cplx x2 = cmul(x, x);
    cplx x3 = cmul(x2, x);
    xpow = cmul(x3, x3);
    break;
   }
   case 8:{
    cplx x2 = cmul(x, x);
    cplx x3 = cmul(x2, x);
    cplx x4 = cmul(x2, x2);
    xpow = cmul(x4, x3);
    break;
   }
   case 9:{
    cplx x2 = cmul(x, x);
    cplx x4 = cmul(x2, x2);
    xpow = cmul(x4, x4);
    break;
   }
  }

  double norm2 = cnorm2(xpow);
  cplx xpow_inv = {xpow.r / norm2, -xpow.i / norm2}; // x^-(d-1)

  double d_inv = 1.0 / (double)deg;
  double coeff = (double)(deg - 1) * d_inv;

  cplx term1 = cscale(x, coeff);
  cplx term2 = cscale(xpow_inv,  d_inv);
  x = cadd(term1, term2);

  conv += 1;
 }
}


static const int palette[10][3] = {
 {255, 64, 64}, {64, 255, 64}, {64, 64, 255},
 {255, 255, 64}, {255, 64, 255}, {64, 255, 255},
 {192, 192, 192}, {255, 128, 64}, {128, 64, 255},
 {64, 128, 255}
};

// computes rows assigned to thread
int
compute_thread(
 void *args
){
 thrd_info_t *ti = (thrd_info_t*) args;
 const int ib = (*ti).ib;
 const int istep = (*ti).istep;
 const int sz = (*ti).sz;
 const int tx = (*ti).tx;
 int_padded *status = (*ti).status;

 for ( int row = ib; row < sz; row += istep ){
  TYPE_ATTR *row_attr = (TYPE_ATTR*)malloc(sz * sizeof(TYPE_ATTR));
  TYPE_CONV *row_conv = (TYPE_CONV*)malloc(sz * sizeof(TYPE_CONV));

  for ( int col = 0; col < sz; ++col ){
   // map pixel to [-2,2]x[-2,2] compex plane based on sz
   double re = -2.0 + 4.0 * ((double)col / (double)(sz - 1));
   double im = 2.0 - 4.0 * ((double)row / (double)(sz - 1));
   cplx z0 = {re, im};

   int conv_cnt = 0;
   int attr = newton_iter(z0, degree, &conv_cnt);

   row_attr[col] = (TYPE_ATTR) attr;
   row_conv[col] = (TYPE_CONV) conv_cnt;
  }
  // update global arrays and update status using mutex to avoid thread overwriting
  mtx_lock(&mtx);
  attractors[row] = row_attr;
  convergences[row] = row_conv;
  status[tx].val = row +istep;
  mtx_unlock(&mtx);
  cnd_signal(&cnd);
 }

 return 0;
}

int
writer_thread(
 void *args
){
 thrd_info_writer_t *wi = (thrd_info_writer_t*) args;
 const int sz = (*wi).sz;
 const int nth = (*wi).nthrds;
 int_padded *status = (*wi).status;
 const int conv_cap = (*wi).conv_cap;

 // name file
 char fname_attr[64], fname_conv[64];
 snprintf(fname_attr, sizeof(fname_attr), "newton_attractors_x%d.ppm", degree);
 snprintf(fname_conv, sizeof(fname_conv), "newton_convergence_x%d.ppm", degree);

 FILE *fa = fopen(fname_attr, "w");
 FILE *fc = fopen(fname_conv, "w");

 // big buffers to reduce number of write calls, only write when buff is full
 setvbuf(fa, NULL, _IOFBF, 1<<20);
 setvbuf(fc, NULL, _IOFBF, 1<<20);

 // write P3 headers
 fprintf(fa, "P3\n%d %d\n255\n", sz, sz);
 fprintf(fc, "P3\n%d %d\n255\n", sz, sz);

 // precompute strings for palette colors
 char pal_str[10][16];
  int pal_len[10];
  for ( int px = 0; px < 10; ++px ) {
   pal_len[px] = snprintf(pal_str[px], sizeof(pal_str[px]), "%d %d %d ", palette[px][0], palette[px][1], palette[px][2]);
  }

 // precompute grayscale strings for normalized values
 int cc = conv_cap;
 char *gray_str = malloc((cc + 1) * 16);
 int *gray_len = malloc((cc + 1) * sizeof(int));
 for ( int gx = 0; gx <= cc; ++gx ) {
  int gray_val = (int)(255.0 * (double)gx / (double) cc + 0.5);
  int pos = gx * 16;
  gray_len[gx] = snprintf(gray_str + pos, 16, "%d %d %d ", gray_val, gray_val, gray_val);
 }


 size_t bufsz = (size_t)sz * 16 + 128; // each pixel has at most 12 chars (we use 16 for good measure + some slack)
 char *buf = malloc(bufsz);

 int write_idx = 0;
 while (write_idx < sz) {
  int ibnd;

  // wait until at least one new row is available to write
  mtx_lock(&mtx);
  for ( ;; ) {
   ibnd = sz;
   for ( int tx = 0; tx < nth; ++tx ) {
    int v = status[tx].val;
    if ( v < 0 ) v = 0;
    if ( ibnd > v ) ibnd = v;
   }
   if ( ibnd <= write_idx ) {
    cnd_wait(&cnd, &mtx);
   }
   else {
    mtx_unlock(&mtx);
    break;
   }
  }

  for ( ; write_idx < ibnd && write_idx < sz; ++write_idx ) {
   TYPE_ATTR *row_attr = attractors[write_idx];
   TYPE_CONV  *row_conv = convergences[write_idx];

   // build attractor row using memcpy of prev precomputed strings
   size_t pos = 0;
   for ( int cx = 0; cx < sz; ++cx ) {
    int a = (int)row_attr[cx];
    if ( a >= degree ) {
     memcpy(buf + pos, "0 0 0 ", 6);
     pos += 6;
    }
    else {
     int l = pal_len[a];
     memcpy(buf + pos, pal_str[a], l);
     pos += l;
    }
   }
   fwrite(buf, 1, pos, fa);
   fwrite("\n", 1, 1, fa);

   // build convergence row using prev precomputed grayscale strings
   pos = 0;
   for (int c = 0; c < sz; ++c) {
    int conv = (int) row_conv[c];
    if (conv > cc)
     conv = cc;
    char *p = gray_str + conv * 16;
    int l = gray_len[conv];
    memcpy(buf + pos, p, l);
    pos += l;
   }
   fwrite(buf, 1, pos, fc);
   fwrite("\n", 1, 1, fc);
  }
 }

 free(buf);
 free(gray_str);
 free(gray_len);

 fclose(fa);
 fclose(fc);

 return 0;
}

static void
parse_args( // requires format -tX -lY d or -lY -tX d as described in assignment
 int argc,
 char **argv
){
 nthrds = 1;
 nmb_lines = 0;

 char *degstr = argv[argc - 1];
 degree = atoi(degstr);
 if (degree <= 0 || degree >= 10)
  exit(1);

 for ( int ix = 1; ix < argc - 1; ++ix ){
  if ( argv[ix][0] == '-' ){
   if ( argv[ix][1] == 't' ){
    nthrds = atoi(argv[ix] + 2);
    if ( nthrds <= 0 )
     nthrds = 1;
   }
   else if ( argv[ix][1] == 'l' ){
    nmb_lines = atoi(argv[ix] + 2);
    if ( nmb_lines <= 0 )
     nmb_lines = 10;
   }
   else
    exit(1);
  }
  else
   exit(1);
 }
}


int
main(
 int argc,
 char **argv
){
 parse_args(argc, argv);
 precompute_roots(degree);
 attractors = (TYPE_ATTR**)malloc(nmb_lines * sizeof(TYPE_ATTR*));
 convergences = (TYPE_CONV**)malloc(nmb_lines * sizeof(TYPE_CONV*));
 for (int ix = 0; ix < nmb_lines; ++ix ){
  attractors[ix] = NULL;
  convergences[ix] = NULL;
 }

 mtx_init(&mtx, mtx_plain);
 cnd_init(&cnd);

 int_padded *status = (int_padded*)malloc(nthrds * sizeof(int_padded));
 for ( int tx = 0; tx < nthrds; ++tx )
  status[tx].val = -1;

 // assign threads to compute function
 thrd_t *cthr = (thrd_t*)malloc(nthrds * sizeof(thrd_t));
 thrd_info_t *cti = (thrd_info_t*)malloc(nthrds * sizeof(thrd_info_t));
 for ( int tx = 0; tx < nthrds; ++tx ){
  cti[tx].tx = tx;
  cti[tx].ib = tx;
  cti[tx].istep = nthrds;
  cti[tx].sz = nmb_lines;
  cti[tx].status = status;
  int r = thrd_create(&cthr[tx], compute_thread, (void*) &cti[tx]);
 }

 // assign threads to write function
 const int conv_cap = 100;
 thrd_t wth;
 thrd_info_writer_t wi;
 wi.sz = nmb_lines;
 wi.nthrds = nthrds;
 wi.status = status;
 wi.conv_cap = conv_cap;
 int rr = thrd_create(&wth, writer_thread, (void*) &wi);

 {
  int res;
  thrd_join(wth, &res);
 }

 for ( int tx = 0; tx < nthrds; ++tx ){
  int r;
  thrd_join(cthr[tx], &r);
}

 free(cthr);
 free(cti);
 free(status);
 free(attractors);
 free(convergences);

 free(roots);

 mtx_destroy(&mtx);
 cnd_destroy(&cnd);

 return 0;
}







