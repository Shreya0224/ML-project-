# AWS Amplify Deployment Guide

## Important

AWS Amplify Hosting can host your frontend files such as `index.html`, `config.js`, `styles.css`, images, and JavaScript.

Your `app.py` file is a Flask backend API. Amplify static hosting does not run a long-lived Flask server directly.

So the correct setup is:

1. Host the frontend on AWS Amplify
2. Host the Flask backend separately on AWS App Runner, Elastic Beanstalk, EC2, or AWS Lambda/API Gateway
3. Put the backend URL inside `config.js`

## Files added for deployment

- `amplify.yml`: tells Amplify how to deploy this static site
- `config.js`: stores the backend API base URL
- `requirements.txt`: backend Python dependencies
- `.gitignore`: keeps local Python files out of git

## Before deploying

1. Make sure your Flask backend is deployed somewhere public
2. Edit `config.js`
3. Replace:

```js
API_BASE_URL: "http://127.0.0.1:5000"
```

with your real backend URL, for example:

```js
API_BASE_URL: "https://your-backend.example.com"
```

## Deploy frontend to Amplify

### Option 1: Deploy from GitHub

1. Push this project to GitHub
2. Open AWS Console
3. Go to `AWS Amplify`
4. Click `New app`
5. Click `Host web app`
6. Choose `GitHub`
7. Authorize GitHub and select your repository
8. Select the branch
9. Amplify will detect `amplify.yml`
10. Click `Save and deploy`

After deployment, Amplify will give you a frontend URL like:

`https://main.xxxxxx.amplifyapp.com`

### Option 2: Drag and drop manual deploy

1. Put all frontend files in one folder:
   - `index.html`
   - `config.js`
   - `styles.css`
   - `script.js`
   - images/videos if used
2. Zip that folder
3. Open `AWS Amplify`
4. Choose manual deploy
5. Upload the zip file

## Backend deployment idea

For the Flask backend, the easiest AWS choices are:

1. `AWS App Runner`
2. `AWS Elastic Beanstalk`

If you want the simplest path, use `AWS App Runner` or `Elastic Beanstalk` for `app.py`.

## Local test before deployment

Run:

```bash
pip install -r requirements.txt
python app.py
```

Then open `index.html` in the browser and confirm the API works locally.

## Recommended final architecture

- Frontend: AWS Amplify
- Backend API: AWS App Runner or Elastic Beanstalk
- Optional domain: Route 53 + custom domain in Amplify
