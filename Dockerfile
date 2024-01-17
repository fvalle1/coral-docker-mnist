FROM debian:buster

RUN apt-get update && apt-get install --yes curl gpg
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install --yes libedgetpu1-std
RUN apt-get install --yes python3-pycoral
RUN apt-get install --yes edgetpu-compiler
RUN apt-get install --yes usbutils
RUN apt-get install --yes python3-pip
RUN python3 -m pip install --no-cache-dir -U pip

COPY requirements.txt /home/data/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /home/data/requirements.txt

ENTRYPOINT ["sh"]
CMD ["/home/data/run.sh"]
