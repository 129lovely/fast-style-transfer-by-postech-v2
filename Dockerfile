# base docker image
FROM tensorflow/tensorflow:1.12.0-gpu-py3

# install tkinter
RUN apt-get update && apt-get install -y python3-tk libgl1-mesa-glx

# install opencv
RUN apt-get -y install wget unzip \
    build-essential cmake git pkg-config libatlas-base-dev gfortran \
    libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
    libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev v4l-utils \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libxvidcore-dev libx264-dev libxine2-dev
RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip && unzip 3.1.0.zip \
    && mv opencv-3.1.0 /opencv
WORKDIR /opencv/3.1.0
RUN cmake -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DWITH_CUDA=OFF \
    -DENABLE_AVX=ON \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=OFF \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DWITH_VTK=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_python2=OFF \
    -DCMAKE_INSTALL_PREFIX=$(python3.5 -c "import sys; print(sys.prefix)") \
    -DPYTHON3_EXECUTABLE=$(which python3.5) \
    -DPYTHON3_INCLUDE_DIR=$(python3.5 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON3_PACKAGES_PATH=$(python3.5 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..
RUN make -j4
RUN make install

# install pip requirements
RUN pip install --upgrade pip
RUN pip install imutils
RUN pip install scipy==1.0.0
RUN pip install numpy
RUN pip install pillow
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install python-dotenv

# change working directory
WORKDIR /app

# switch to a non-root user
RUN useradd ponix && chown -R ponix /app
USER ponix

CMD ["/bin/bash"]