# CV1

## Members:
- Yao Meng
- Chen Xue
- Tiancheng Hu

## Starts everything from scratch
- Pycharm 
  - new Project
  - Get from VCS
  - login Github 
#- cd into the desired directory, and <pre> git init</pre>
- first <pre> sudo apt install git-all</prep>
- git clone this repo to your local direction first, still in the desired directory and type
  - <prep>git clone -b master https://github.com/Aniccam/CV1.git </prep>
- download Pycharm, miniconda3
#- (in Pycharm): create virtual environment better with conda
  #- click <pre> New Project </pre>
  #- select the former directory
  #- leave everything as it was, in conda env only change python version in 3.8, name one env name as you like
                
## How to install dependencies:
1. cd to the directory where requirements.txt is located
2. activate your virtual environment (conda activate XXX)
3. run this in the activated env: <pre> pip install -r requirements.txt</pre>
Or manual operate in Pycharm- Settings- Python Interpreter- Conda


## Once started  
- localize changes from remote repo  <pre>git pull </pre>                 
- don't ignore the dot in the end, this will put your changes to be repared <pre>git add . </pre>                 
- write some commitment to your code <pre>git commit </pre>               
- Time to save changes! <pre>ctrl + X </pre>                
- update changes to remote repo <pre>git push </pre>  


## Most Important Rules During Semester:
- we do specific part assignments, you can read other's code but only edit your own part, fix conflict can be time-consuming 

- "pull" before anything or anytime starts programmming 

- "push" after finishing coding 

