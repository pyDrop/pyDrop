# Git Cheatsheet

## Some basic get commands to remember

`git pull` - pull work in git to your local directory

`git push` - push local commits to git

### Working with branches

`git checkout -b <branch_name>` - create a new branch locally

`git branch` - list branches

`git branch -a` - list all branches that exist

`git checkout <branch_name>` - check out an existing branch

### Maintaining a branch

`git add <filename>` - add a file to version control

`git commit` - commit any new changes to version control

#### Basic commit guidelines

Being intentional about when you commit is important. Committing after every single change you make, or every line of code you write, is excessive. However, you don't want to wait until your project is finished to make your first commit. You want to commit your changes in bite sized chunks that make it easy to see and understand your progress over time. For example, if you have two functions that you are trying to change, you might want to make changes to one function first, commit those changes, and then make changes to the other function. If those changes go hand in hand, however, you might want to put them in the same commit. 

You also want to be mindful of your commit messages. They should clearly describe the change made. When you run `git log` you will see all of the changes in your history. When you look at this, you want to be able to quickly understand the changes you have made, and the changes other people have made. 

## Making a pull request
When you think you are finished with your work and/or want to see it in git, execute the command `git push`. If this is your first push, git will complain, but tell you which command to run. 

To make a pull request, navigate to git in your browser. A band will appear that tells you that you have recently pushed code, along with a button suggesting you make a pull request. Make sure to give it a descriptive title. In the description section below, described the changes you have made, and note anything that is important to the reader. 

Before requesting review, make sure you scroll through your code in git. That this is a good opportunity for you to catch any formatting errors and other mistakes. When you are satisfied with the quality of the code in your pull request, look at the reviewers section on the right and click on the people you want to review your code. The people you select will get an email saying that their review has been requested.

When Seth and Kari have reviewed your code, we ask that you address and resolve every single comment we make. Once this is done, click the button to re-request our review. When you have gotten your final approval, feel free to merge.

### Merging your pull request

When you merge your pull request, we ask that you use the `Squash and Merge` option. When you click `Squash and Merge`, a text box will appear below that contains all of your commit history on that branch. Please edit that textbox to contain a neat, bulleted list that concisely describes the changes you made in your PR. You can now click `Squash and Merge` to merge your branch.

### Rebasing

With multiple people working on one project, it is common that your branch will be out of date with main. When this happens, you can rebase your branch to update it with the changes that are in main.

For example, this is what you would do to rebase your branch onto main:

`git rebase -i main` - the `-i` tells git that this is an interactive rebase

Follow the examples that git gives you. Be mindful of which changes are important to keep from the main branch, and which are important to keep from your branch.

After a rebase, you will likely have to force push your changes with: `git push -f`

## Further Reading

[git book](https://git-scm.com/book/en/v2)
