# Author Identification Project (AIP)

## How to run our code

### Requirements
We used Python 3.9.6, so my code is set up to work with this version.
The required modules are listed in `requirements.txt`, and can be installed by running the command `pip install -r requirements.txt`.

### Flask web server
To run our Streamlit app directly on your computer, `cd` into the root directory (`aip/`) and run the following command: 
```bash
$ python3 -m streamlit run app.py
```
Then, go to [http://localhost:8501](http://localhost:8501/) in a browser to access the website and use the text area to input text, or the file upload button to upload a .txt file. Click the submit button to process the text and view the results. 
If your text was an exact match to a text in our database, the matcehed book(s) will be displayed. Otherwise, a prediction made by our classification model will be displayed, along with an option to add the entry to our database (so that next time it is searched, it will be an exact match).

### Docker image and container
To create a Docker image of this Streamlit app, use the `Dockerfile`. To do this, make sure Docker is installed, and then `cd` into this directory (`aip/`) and run the following command:
```bash
$ docker build --tag aip .   
```
Then, to run the Docker container using this image, run the following command: 
```bash
$ docker run -dp 8501:8501 aip
```
You can then access the Streamlit app at [http://localhost:8501](http://localhost:8501/).
Note: if you're running the server from Docker and are using the "upload" button to upload a .txt file, this .txt file has to be in the Docker container.

![image](https://user-images.githubusercontent.com/45803348/229929185-b503c719-a317-4b7a-b0f5-0a039944cbe4.png)
