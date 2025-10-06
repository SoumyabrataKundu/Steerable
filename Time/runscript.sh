
channels=(1 2 4 8)
kernels=(5)
n_radius=(1 2 3 4)
max_m=(0 1 2 3)
conv_first=(0 1)

wait_for_jobs() {
    while [ $(squeue -u $USER | tail -n +2 | wc -l) -ge 28 ]; do
        sleep 10
    done
}

for channel in "${channels[@]}"
do
    for kernel in "${kernels[@]}"
    do
        for radius in "${n_radius[@]}"
        do
            for k in "${max_m[@]}"
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

                    wait_for_jobs
                    sbatch script_temp.sh
                    rm script_temp.sh

                done
                echo "Submitted Job -- Channel : ${channel}  Kernel : ${kernel}  Radius : ${radius}  Cutoff : ${k}"

            done
            echo 

        done
    done
done

