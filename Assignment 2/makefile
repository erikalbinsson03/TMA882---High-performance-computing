CC = gcc
CFLAGS = -O2 -fopenmp
LDFLAGS = -lm

BINS = distances

.PHONY: all
all: $(BINS)

$(BINS): distances.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(BINS) *.o

