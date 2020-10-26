#!/bin/bash

# This script is meant to be called in the "deploy" step defined
# in .circleci/config.yml. See https://circleci.com/docs/2.0 for more details.

# We have three possibily workflows:
#   If the git branch is 'master' then we want to commit and merge the dev/
#       docs on gh-pages
#   If the git branch is 'advanced' then we want to commit and merge the
#       advanced docs on gh-pages
#   If the git branch is [0-9].[0.9].X (i.e. 0.9.X, 1.0.X, 1.2.X, 41.21.X) then
#        we want to commit and merge the major.minor/ docs on gh-pages
#   If the git branch is anything else then we just want to test that committing
#       the changes works so that any issues can be debugged

function doc_clone_commit {
    # Note that we use [skip ci] to tell CircleCI not to build the commit
    MSG="Updating the docs in $DIR/ for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1 [skip ci]"

    cd $HOME
    git clone --depth 1 --no-checkout "git@github.com:"$ORGANIZATION"/"$DOC_REPO".git";
    cd $DOC_REPO
    git reset --hard origin/$DOC_BRANCH
    git rm -rf $DIR/ && rm -rf $DIR/
    cp -R $HOME/project/doc/_build/html/stable $DIR
    git config --global user.email $EMAIL
    git config --global user.name $USERNAME
    git config --global push.default matching
    git add -f $DIR/
    git commit -m "$MSG" $DIR
}

# Test that the vars have been set
if [ -z ${CIRCLE_BRANCH+x} ]; then echo "CIRCLE_BRANCH is unset"; fi
if [ -z ${CIRCLE_SHA1+x} ]; then echo "CIRCLE_SHA1 is unset"; fi
if [ -z ${CIRCLE_REPOSITORY_URL+x} ]; then echo "CIRCLE_REPOSITORY_URL is unset"; fi
if [ -z ${CIRCLE_PROJECT_REPONAME+x} ]; then echo "CIRCLE_PROJECT_REPONAME is unset"; fi
if [ -z ${HOME+x} ]; then echo "HOME is unset"; fi
if [ -z ${EMAIL+x} ]; then echo "EMAIL is unset"; fi
if [ -z ${USERNAME+x} ]; then echo "USERNAME is unset"; fi

# Determine which of the three workflows to take
if [ "$CIRCLE_BRANCH" = "master" ]
then
    # Changes are made to dev/ directory
    DIR="ramp-workflow/dev"
    doc_clone_commit
    git push origin $DOC_BRANCH
    echo "Push complete"
elif [ "$CIRCLE_BRANCH" = "pull/254" ]
then
    # Changes are made to advanced/ directory
    DIR="ramp-workflow/advanced"
    doc_clone_commit
    echo $USERNAME
    git push origin $DOC_BRANCH
    echo "Push complete"
elif [[ "$CIRCLE_BRANCH" =~ ^[0-9]+\.[0-9]+\.X$ ]]
then
    # Strip off .X from branch name, so changes will go to 0.1/, 91.235/, etc
    DIR="ramp-workflow/${CIRCLE_BRANCH::-2}"
    doc_clone_commit
    git push origin $DOC_BRANCH
    echo "Push complete"
else
    DIR="ramp-workflow/dev"
    doc_clone_commit
    echo "Test complete, changes NOT pushed"
fi
