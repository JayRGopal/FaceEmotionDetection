mkdir -p ~/NAS/Conda_Backups/klab && conda env list | awk '{print $1}' | grep -v "#" | xargs -I {} conda env export -n {} > ~/NAS/Conda_Backups/klab/{}.yml
