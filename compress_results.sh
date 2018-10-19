#!/usr/bin/env bash

# Original folder where models lay
export RESULTS_FOLDER=results

# Temporary folder to be compressed
export EXPORTED=`date +%Y%m%d%H%M%S`_exported_results
mkdir $EXPORTED

# Copying the folder and filtering the extracted
shopt -s extglob
cp -R --parents $RESULTS_FOLDER/*/!(*.h5) $EXPORTED

# Compressing the content of the folder
tar -czvf $EXPORTED.tar.gz $EXPORTED/*

# Removing the temporary folder
rm -Rf $EXPORTED/