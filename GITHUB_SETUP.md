# Push this project to GitHub

The repo is initialized with one commit. To put it on GitHub:

1. **Create a new repository on GitHub**
   - Go to [github.com/new](https://github.com/new)
   - Name it e.g. `movie_recommendation_system`
   - Do **not** add a README, .gitignore, or license (this project already has them)
   - Create the repository

2. **Add the remote and push**

   ```bash
   cd /Users/rhou/cs_projects/movie_recommendation_system
   git remote add origin https://github.com/YOUR_USERNAME/movie_recommendation_system.git
   git branch -M main
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` with your GitHub username. If you use SSH:

   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/movie_recommendation_system.git
   git push -u origin main
   ```

3. **After cloning elsewhere**: place `movies_metadata.csv` in the project root (see main README).
