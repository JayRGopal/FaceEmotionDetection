mkdir -p /mnt/NAS/Conda_Backups/klab && conda env list | awk '{print $1}' | grep -v "#" | xargs -I {} conda env export -n {} > /mnt/NAS/Conda_Backups/klab/{}.yml
