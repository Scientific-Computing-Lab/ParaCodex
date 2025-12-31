SHELL=/bin/sh
BENCHMARK=bt
BENCHMARKU=BT
VEC=

include config/make.def

include sys/make.common

OBJS = bt.o exact_solution.o set_constants.o verify.o \
       solve_subs.o print_results.o c_timers.o wtime.o  
#       z_solve$(VEC).o add.o error.o verify.o \
#       ${COMMON}/print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o


# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h



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


bt.o:             bt.c  header.h npbparams.h
exact_solution.o: exact_solution.c  header.h npbparams.h
set_constants.o:  set_constants.c  header.h npbparams.h
solve_subs.o:     solve_subs.c  npbparams.h
verify.o:         verify.c  header.h npbparams.h

clean:
	- rm -f *.o *~ mputil* ../common/*.o *.cu *.ptx *.w2c.c *.w2c.h *.x *.i *.t *.spin *.B
	- rm -f  npbparams.h core

.PHONY: run
run: ${PROGRAM}
	./${PROGRAM}
