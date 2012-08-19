for pkg in g2o Pangolin Sophus VisionTools; do
    if [ ! -x $pkg/build ]; then
        for p in `ls ${pkg}_*.patch`; do
            patch -d ${pkg} -p1 < $p
        done
        mkdir -p $pkg/build
        pushd $pkg/build
        if [ "$pkg" = "Pangolin" ]; then
            cmake .. -DCMAKE_INSTALL_PREFIX:PATH=.. -DBUILD_SHARED_LIBS=ON
        else
            cmake .. -DCMAKE_INSTALL_PREFIX:PATH=..
        fi
    else
        pushd $pkg/build
    fi
    make -j4
    if [ ! "$pkg" = "VisionTools" ]; then
        make install
    fi
    popd >/dev/null
done


