#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>

typedef struct { int16_t x, y, z; } point_t;

#define MEMORY_LIM_BYTES (5 * 1024 * 1024)

// deconstruct a character of form '±ii.ddd' and convert to integer ±iiddd
int
parse_coord(
 const char *s
){
 int sign = (s[0] == '-') ? -1 : 1;
 int i2 = s[1] - '0';
 int i1 = s[2] - '0';
 int d1 = s[4] - '0';
 int d2 = s[5] - '0';
 int d3 = s[6] - '0';
 int val = i2 * 10000 + i1 * 1000 + d1 * 100 + d2 * 10 + d3;
 return sign * val;
}


uint32_t count_rows(
 const char *fname
){
 FILE *f = fopen(fname, "r");

 char buf[32]; // arbitrary buffer larger than characters in row (>27?)
 uint64_t n = 0;
 while ( fgets(buf, sizeof(buf), f) )
  n += 1;

 fclose(f);

 return n;
}


int
main(
 int argc,
 char **argv
){

 int num_threads = 1;
 // read input -t argument
 for ( int i = 1; i < argc; ++i ){
  if ( argv[i][0] == '-' && argv[i][1] == 't' ){
   if ( argv[i][2] )
    num_threads = atoi(&argv[i][2]);
  else if ( i + 1 < argc )
   num_threads = atoi(argv[++i]);
  }
 }
 if ( num_threads < 1)
  num_threads = 1;
 omp_set_num_threads(num_threads);


 const char *fname = "cells";
 uint32_t N = count_rows(fname);


// decide adaptive block memory size
 size_t hist_bytes = 3500 * sizeof(uint32_t) * (size_t)num_threads; // 3465 max number of distances
 size_t reserve = 64 * 1024;
 size_t available = MEMORY_LIM_BYTES - reserve - hist_bytes;
 size_t block_points = available / (2 * 6); // 2 blocks and 6 bytes per point (3*int16_t)

 if ( block_points < 1 )
  block_points = 1;
 if ( block_points > (size_t)N )
  block_points = (size_t)N;
 size_t num_blocks = (size_t)((N + block_points - 1) / block_points);


// record block offsets/block positions
 off_t *offsets = malloc(num_blocks * sizeof(off_t));
 FILE *f = fopen(fname, "r");
 char line[32];

 size_t bi = 0;
 while ( bi < num_blocks ){
  offsets[bi++] = ftello(f);
  size_t ix;
  for ( ix = 0; ix < block_points; ++ix ){
   if ( fgets(line, sizeof(line), f) == NULL )
    break;
  }
  if ( ix < block_points )
   break;
 }
 num_blocks = bi;


// two buffer blocks containing block_points points to compare
 point_t *bufA = malloc(block_points * sizeof(point_t));
 point_t *bufB = malloc(block_points * sizeof(point_t));

// define cumulative global and local histogram for each thread to avoid overwriting
 uint32_t *all_hist = calloc((size_t)(num_threads * 3500), sizeof(uint32_t));
 uint32_t *global_hist = calloc(3500, sizeof(uint32_t));


// main loops
 for ( size_t bi = 0; bi < num_blocks; ++bi ){
  fseeko(f, offsets[bi], SEEK_SET); // set pointer to correct block
  size_t na = 0;

  while ( na < block_points && fgets(line, sizeof(line), f)){ // load in block to bufA
   const char *s = line;
   int xi = parse_coord(s);
   s += 8;
   int yi = parse_coord(s);
   s += 8;
   int zi = parse_coord(s);
   bufA[na].x = (int16_t)xi;
   bufA[na].y = (int16_t)yi;
   bufA[na].z = (int16_t)zi;
   na += 1;
  }
  #pragma omp parallel
  {
   int thread_id = omp_get_thread_num();
   uint32_t *local = all_hist + (size_t)(thread_id * 3500);

   #pragma omp for schedule(dynamic)
   for ( size_t p = 0; p < na; ++p ) // compute distance and increment histogram for pairs within bufA
    for ( size_t q = p + 1; q < na; ++q ){
     double dx = ((int)bufA[p].x - (int)bufA[q].x) / 1000.0;
     double dy = ((int)bufA[p].y - (int)bufA[q].y) / 1000.0;
     double dz = ((int)bufA[p].z - (int)bufA[q].z) / 1000.0;
     double d = sqrt(dx * dx + dy * dy + dz * dz);
     int idx = (int)(d * 100.0 + 0.5); // round up to nearest integer containing two decimals

     local[idx]++; // increment in corresponding thread histogram
    }
  }

// load in block to bufB
  for ( size_t bj = bi +1; bj < num_blocks; ++bj ){
   fseeko(f, offsets[bj], SEEK_SET);
   size_t nb = 0;
   while ( nb < block_points && fgets(line, sizeof(line), f)){
    const char *s = line;
    int xi = parse_coord(s);
    s += 8;
    int yi = parse_coord(s);
    s += 8;
    int zi = parse_coord(s);
    bufB[nb].x = (int16_t)xi;
    bufB[nb].y = (int16_t)yi;
    bufB[nb].z = (int16_t)zi;
    nb += 1;
   }

   #pragma omp parallel
   {
    int thread_id = omp_get_thread_num();
    uint32_t *local = all_hist + (size_t)(thread_id * 3500);

   #pragma omp for schedule(dynamic) // pair-wise compute distance between points in A to B
    for ( size_t p = 0; p < na; ++p )
     for ( size_t q = 0; q < nb; ++q ){
      double dx = ((int)bufA[p].x - (int)bufB[q].x) / 1000.0;
      double dy = ((int)bufA[p].y - (int)bufB[q].y) / 1000.0;
      double dz = ((int)bufA[p].z - (int)bufB[q].z) / 1000.0;
      double d = sqrt(dx * dx + dy * dy + dz * dz);
      int idx = (int)(d * 100.0 + 0.5);

      local[idx]++;
     }
   }
  }
 }

 fclose(f);

 for ( size_t tx = 0; tx < num_threads; ++tx ){ // add to combined histogram
  uint32_t *local = all_hist + tx * 3500;
  for ( int kx = 0; kx < 3500; ++kx )
   global_hist[kx] += local[kx];
 }

 for ( int kx = 0; kx < 3500; ++kx ){
  if ( global_hist[kx] > 0)
   printf("%05.2f %u\n", kx / 100.0, (unsigned)global_hist[kx]);
 }



 free(offsets);
 free(bufA);
 free(bufB);
 free(all_hist);
 free(global_hist);

 return 0;
}
