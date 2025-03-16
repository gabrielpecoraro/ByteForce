# ByteForce Datascience Project

## Prerequisites
Make sure you have the following installed on your system:
- Git
- A GitHub account

## Step 1: Fork the Repository
1. Navigate to the GitHub repository you want to contribute to.
2. Click on the `Fork` button in the top-right corner.
3. GitHub will create a copy of the repository in your account.

## Step 2: Clone the Forked Repository
1. Go to your GitHub profile and navigate to the forked repository.
2. Click on the `Code` button and copy the repository URL.
3. Open a terminal and run the following command:
   ```sh
   git clone <repository-url>
   ```
   Replace `<repository-url>` with the URL you copied.
4. Change into the repository directory:
   ```sh
   cd <repository-name>
   ```

## Step 3: Set Up the Upstream Repository
1. Add the original repository as an upstream remote:
   ```sh
   git remote add upstream <original-repository-url>
   ```
2. Verify the remote repositories:
   ```sh
   git remote -v
   ```

## Step 4: Create a New Branch for a Task
1. Fetch the latest changes from the upstream repository:
   ```sh
   git fetch upstream
   ```
2. Ensure you're on the main branch:
   ```sh
   git checkout main
   ```
3. Merge any latest changes:
   ```sh
   git merge upstream/main
   ```
4. Create a new branch for your task:
   ```sh
   git checkout -b <branch-name>
   ```
   Replace `<branch-name>` with a meaningful name related to the task (e.g., `feature-login`, `fix-bug-123`).

## Step 5: Work on the Task and Commit Changes
1. Make the necessary changes in your branch.
2. Stage the changes:
   ```sh
   git add .
   ```
3. Commit the changes:
   ```sh
   git commit -m "Describe the changes made"
   ```

## Step 6: Push the Branch to Your Fork
1. Push your branch to GitHub:
   ```sh
   git push origin <branch-name>
   ```

## Step 7: Create a Pull Request
1. Go to your GitHub repository.
2. Click on `Compare & pull request` for the new branch.
3. Add a title and description for the changes.
4. Click `Create pull request`.
5. Wait for a review and merge.

## Step 8: Keep Your Fork Updated
To avoid conflicts, regularly sync your fork with the upstream repository:
1. Fetch the latest changes:
   ```sh
   git fetch upstream
   ```
2. Merge them into your main branch:
   ```sh
   git checkout main
   git merge upstream/main
   ```
3. Push the updated main branch to your fork:
   ```sh
   git push origin main
   ```

