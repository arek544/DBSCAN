FROM python:3.8.7

# creating an environment variable that holds the project directory
ENV PYTHONPATH="/work"
WORKDIR "${PYTHONPATH}"

COPY requirements.txt requirements.txt

# installing the necessary dependencies for the operation of the project
# RUN sudo apt-get update 
RUN pip install -r requirements.txt 
RUN pip install jupyterlab numpy pandas sklearn matplotlib seaborn
RUN echo "alias jp=\"jupyter lab --no-browser --allow-root --ip=0.0.0.0\""  >> ~/.bashrc

# comment this to turn off autostart of jupyter lab
CMD ["jupyter", "lab", "--no-browser", "--allow-root", "--ip=0.0.0.0"]