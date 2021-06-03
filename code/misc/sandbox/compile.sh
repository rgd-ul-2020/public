#!/bin/bash

mkdir -p build
cd build

case $1 in
    clean)
    	rm -R *
	;;
    *)
        CMAKE_GEN="Unix Makefiles"
        HAVE_INT64=

        #IF WINDOWS/MSYS
        if [ -n "`uname -s | grep -i mingw`" ]; then
            CMAKE_GEN="MSYS Makefiles"

            #IF WIN64/MSYS
            if [ -n "`uname -s | grep -i mingw64`" ]; then
                echo ""
                HAVE_INT64="-DHAVE_LONG_LONG_INT_64=1"
            fi
        fi

        cmake -G "${CMAKE_GEN}" ${HAVE_INT64} ..
        make
	;;
esac

