FROM python:2.7

RUN mkdir app

WORKDIR app 

COPY . . 

RUN pip install -r requirements.txt 

RUN make -C  darknet

CMD ["/bin/bash/"]/
