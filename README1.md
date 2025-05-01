This is the project about question paper generation using large languagae models

Here we are using Llama2 model to generate the question.

to run this project


create a new folder named model->
Go the link given below and download the model and add the downloaded file in model folder.

https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin



add the database qpgen_subjectspdfs.sql to MySQL database with any name (prefer 'QPGen' as name ) (go through this link to how to add the database https://www.youtube.com/watch?v=7Cbm5vPQvNI)

goto app.py and give the password of your MySQL database.

create conda environment

conda create -n QPGen

then activate it using the code

conda activate

install python 3.12.8 version

conda install python==3.12.8

install all the requirements

pip install -r requirements.txt

open your xampp and start the apache server and mysql

run the code project using

python app.py
