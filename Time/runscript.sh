
channels=(2 4 8 16 32 64 128)
kernels=(2 3 4 5 6 7)
n_radius=(2 4 6 8 10)
max_m=(4 8 12 16)
batch_size=10
n_simulations=10



for channel in "${channels[@]}"
do
    for kernel in "${kernels[@]}"
    do
        for radius in "${n_radius[@]}"
        do
            for k in "${max_m[@]}"
            do
                sed -i "s/CHANNEL/${channel}/g" script.sh
                sed -i "s/KERNEL/${kernel}/g" script.sh
                sed -i "s/RADIUS/${radius}/g" script.sh
                sed -i "s/MAXM/${k}/g" script.sh

                sbatch script.sh
            done
        done
    done
done