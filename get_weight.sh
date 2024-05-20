#!/bin/bash


#Public Nextcloud URL
wget https://owncloud.ut.ee/owncloud/s/dn2t2NzgAaq6dXz/download/weights.zip

unzip weights.zip

# Clean up the zip file
rm "results.zip"

echo "Download and extraction complete."
