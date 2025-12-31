#include <stdio.h>      

#include <stdlib.h>     

#include <string.h>     

#include <chrono>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))



#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}



#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}


unsigned int mixRemainder(unsigned int a, 
    unsigned int b, 
    unsigned int c, 
    unsigned int k0,
    unsigned int k1,
    unsigned int k2,
    unsigned int length ) 
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;              

  }

  final(a,b,c);
  return c;
}

unsigned int hashlittle( const void *key, size_t length, unsigned int initval)
{
  unsigned int a,b,c;                                          


  

  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  const unsigned int *k = (const unsigned int *)key;         


  

  while (length > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a,b,c);
    length -= 12;
    k += 3;
  }

  

  


  switch(length)
  {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;              

  }

  final(a,b,c);
  return c;
}


int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <block size> <number of strings> <repeat>\n", argv[0]);
    return 1;
  }

  int block_size = atoi(argv[1]);  

  unsigned long N = atol(argv[2]); 

  int repeat = atoi(argv[3]);

  

  const char* str = "Four score and seven years ago";
  unsigned int c = hashlittle(str, 30, 1);
  printf("input string: %s hash is %.8x\n", str, c);   


  unsigned int *keys = NULL;
  unsigned int *lens = NULL;
  unsigned int *initvals = NULL;
  unsigned int *out = NULL;

  

  posix_memalign((void**)&keys, 1024, sizeof(unsigned int)*N*16);
  posix_memalign((void**)&lens, 1024, sizeof(unsigned int)*N);
  posix_memalign((void**)&initvals, 1024, sizeof(unsigned int)*N);
  posix_memalign((void**)&out, 1024, sizeof(unsigned int)*N);

  

  srand(2);
  char src[64];
  memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)keys+i*16*sizeof(unsigned int), src, 64);
    lens[i] = rand()%61;
    initvals[i] = i%2;
  }

  auto start = std::chrono::steady_clock::now();

    {
    for (int n = 0; n < repeat; n++) {
            for (unsigned long id = 0; id < N; id++) {
        unsigned int length = lens[id];
        const unsigned int initval = initvals[id];
        const unsigned int *k = keys+id*16;  


        

        unsigned int a,b,c; 
        unsigned int r0,r1,r2;
        a = b = c = 0xdeadbeef + length + initval;

        

        while (length > 12) {
          a += k[0];
          b += k[1];
          c += k[2];
          mix(a,b,c);
          length -= 12;
          k += 3;
        }
        r0 = k[0];
        r1 = k[1];
        r2 = k[2];

        

        

        out[id] = mixRemainder(a, b, c, r0, r1, r2, length);
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

  printf("Verify the results computed on the device..\n");
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&keys[i*16], lens[i], initvals[i]);
    if (out[i] != c) {
      printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i, out[i], c);
      error = true;
      break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  free(keys);
  free(lens);
  free(initvals);
  free(out);

  return 0;
}