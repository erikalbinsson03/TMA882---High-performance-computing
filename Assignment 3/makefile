BINS = newton
CFLAGS = -std=c11 -O3 -pthread


.PHONY : all
all : $(BINS)

newton : newton.c
	gcc $(CFLAGS) -lm -o $@ $<

.PHONY : clean
clean :
	rm -rf $(BINS)
