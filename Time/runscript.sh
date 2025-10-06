
channels=(2 4 8 16 32 64)
kernels=(2 3 4 5)
n_radius=(2 4 6 8)
max_m=(4 8 12 16)
conv_first=(0 1)

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

                    JOB_ID=$(sbatch script_temp.sh | awk '{print $NF}')
                    rm script_temp.sh

                done
                echo "Submitted Job -- Channel : ${channel}  Kernel : ${kernel}  Radius : ${radius}  Cutoff : ${k}"

            done
            while squeue -j "$JOB_ID"| grep "$JOB_ID" > /dev/null 2>&1; do
                sleep 10
            done
            echo 

        done
    done
done

