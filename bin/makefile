include ../lib/config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

CXX = g++
CFLAGS_SPAMS = -O3 -mtune=native -fomit-frame-pointer -Wall -fopenmp
LDFLAGS_SPAMS = -O -lblas -llapack -lgomp

TARGETS = bbffmt msermatch siftmatch bbfcreate bbfdetect swtcreate swtdetect dpmcreate dpmdetect convert tld icfcreate icfdetect icfoptimize cifar-10 image-net cnnclassify dpmsparsedetect

SPAMS_TARGETS = traindl omp
SPAMS_HEADERS = cblas_alt_template.h cblas_defvar.h decomp.h dicts.h fista.h linalg.h list.h misc.h project.h shared.hpp spams.h utils.h

all: libccv.a $(TARGETS) $(SPAMS_TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm -f *.o $(TARGETS) $(SPAMS_TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
	
spamsclean:
	rm -f *.o $(SPAMS_TARGETS)

$(SPAMS_TARGETS): %: %.o
	$(CXX) $< -o $@ $(LDFLAGS_SPAMS)

%.o: %.cpp $(SPAMS_HEADERS)
	$(CXX) $(CFLAGS_SPAMS) -c $<
