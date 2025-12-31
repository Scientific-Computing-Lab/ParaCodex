SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=lu-serial
VEC=

include config/make.def

include sys/make.common

OBJS = lu.o read_input.o \
       domain.o setcoeff.o setbv.o exact.o setiv.o \
       calcnp.o pintgr.o error.o verify.o \
       ${COMMON}/print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o


# npbparams.h is included by applu.incl
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# applu.incl: npbparams.h

${PROGRAM}: config
	@if [ x$(VERSION) = xvec ] ; then	\
		${MAKE} VEC=_vec exec;		\
	elif [ x$(VERSION) = xVEC ] ; then	\
		${MAKE} VEC=_vec exec;		\
	else					\
		${MAKE} exec;			\
	fi

exec: $(OBJS)
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

lu.o:		lu.c applu.incl npbparams.h
calcnp.o:   calcnp.c applu.incl npbparams.h
error.o:	error.c applu.incl npbparams.h
exact.o:	exact.c applu.incl npbparams.h
pintgr.o:	pintgr.c applu.incl npbparams.h
read_input.o:	read_input.c applu.incl npbparams.h
setbv.o:	setbv.c applu.incl npbparams.h
setiv.o:	setiv.c applu.incl npbparams.h
setcoeff.o:	setcoeff.c applu.incl npbparams.h
domain.o:	domain.c applu.incl npbparams.h
verify.o:	verify.c applu.incl npbparams.h

clean:
	- /bin/rm -f npbparams.h
	- /bin/rm -f *.o *~ ../common/*.o *.w2c.cu *.w2c.h *.w2c.c *.i *.B *.spin *.t *.s *.gpu *.x *.ptx *.n00*.h

#===============================================================================
# Correctness checking targets
#===============================================================================

GATE_ROOT ?= /root/codex_baseline/serial_omp_nas_workdir
REF_DIR   ?= $(GATE_ROOT)/golden_labels/src
REF_BIN   ?= $(REF_DIR)/$(BENCHMARKU)/$(BENCHMARK).$(CLASS).x
REF_MAKE  ?= Makefile
RUN_ARGS  ?=
LAUNCHER  ?=

.PHONY: run
run: ${PROGRAM}
	$(LAUNCHER) ./${PROGRAM} $(RUN_ARGS)

.PHONY: ref_build
ref_build:
	$(MAKE) -C $(REF_DIR)/$(BENCHMARKU) -f $(REF_MAKE) clean
	$(MAKE) -C $(REF_DIR)/$(BENCHMARKU) -f $(REF_MAKE) \
	    CC="$(CC)" CLASS="$(CLASS)" CCC="$(CCC)" \
	    CFLAGS="$(CFLAGS)" CLINKFLAGS="$(CLINKFLAGS)"

.PHONY: check-correctness
check-correctness: ${PROGRAM} ref_build
	OMP_TARGET_OFFLOAD=MANDATORY \
	python3 $(GATE_ROOT)/gate_sdk/scripts/gate_harness.py $(REF_BIN) ./${PROGRAM} $(RUN_ARGS) 
