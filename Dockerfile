FROM python:3.9

WORKDIR /usr/src/app

RUN pip install --no-cache-dir  lungdata

COPY hello.py .

ENTRYPOINT [ "python", "hello.py" ]
