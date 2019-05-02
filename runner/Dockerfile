FROM python:3.6
ADD . /
RUN pip install -r requirements.txt
RUN pip install -e wazuhl-gym
WORKDIR /notebooks

#ENTRYPOINT [ "python", "./main.py" ]
CMD jupyter notebook --ip 0.0.0.0 --port 1303 --no-browser --allow-root