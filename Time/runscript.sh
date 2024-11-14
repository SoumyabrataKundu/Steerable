
channels=(8)
kernels=(5)
n_radius=(2)
max_m=(4 8 12 16)
restricted=(0 1)
conv_first=(0 1)

batch_size=5
n_simulations=10



for channel in "${channels[@]}"
do
    for kernel in "${kernels[@]}"
    do
        for radius in "${n_radius[@]}"
        do
            for k in "${max_m[@]}"
            do
                for restrict in "${restricted[@]}"
                do
                    for conv in "${conv_first[@]}"
                    do
                        cp script.sh script_temp.sh

                        sed -i "s/CHANNEL/${channel}/g" script_temp.sh
                        sed -i "s/KERNEL/${kernel}/g" script_temp.sh
                        sed -i "s/RADIUS/${radius}/g" script_temp.sh
                        sed -i "s/MAXM/${k}/g" script_temp.sh
                        sed -i "s/RESTRICTED/${restrict}/g" script_temp.sh
                        sed -i "s/CONVFIRST/${conv}/g" script_temp.sh

                        sbatch script_temp.sh
                        rm script_temp.sh

                    done
                done
            done
        done
    done
done
