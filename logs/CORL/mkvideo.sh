rm -r imgs
mkdir imgs
python3 video_gen.py june_14_trial/real/star_pid_0.npz june_14_trial/real/star_mppi_0.8.npz  june_14_trial/real/star_ppo_0.8.npz -tt 4.2 --hovertime 4.0 --runtime 12.0 -bh 0.6
ffmpeg -framerate 50 -i imgs/%04d.png -pix_fmt yuv420p videos/star_ppo_pid_mppi.mp4 -y

rm -r imgs
mkdir imgs
python3 video_gen.py june_14/real/gui_pid_0.npz june_14/real/gui_mppi_0.npz june_14/real/gui_ppo_nfb_0.npz -tt 4.2 --hovertime 4.0 --runtime 12.0 -bh 0.6
ffmpeg -framerate 50 -i imgs/%04d.png -pix_fmt yuv420p videos/gui_RL_ppo_pid_mppi.mp4 -y

rm -r imgs
mkdir imgs
python3 video_gen.py june_14_trial/real/star_pid_L1_0.8.npz june_14_trial/real/star_mppi_L1_0.8.npz june_21/real/star_ppo_L1_nfb_final_2.npz -tt 4.2 --hovertime 4.0 --runtime 12.0 -bh 0.6
ffmpeg -framerate 50 -i imgs/%04d.png -pix_fmt yuv420p videos/star_ppo_pid_mppi_L1.mp4 -y

rm -r imgs
mkdir imgs
python3 video_gen.py june_14_trial/real/wind_star_pid_L1_0.8.npz june_14_trial/real/wind_star_mppi_L1_0.8.npz june_21/real/wind_star_ppo_L1_nfb_final.npz -tt 4.2 --hovertime 4.0 --runtime 12.0 -bh 0.6
ffmpeg -framerate 50 -i imgs/%04d.png -pix_fmt yuv420p videos/wind_star_ppo_pid_mppi_L1.mp4 -y

rm -r imgs
mkdir imgs
python3 video_gen.py june_21/real/NW_pid_nofb.npz june_21/real/NW_mppi_nofb.npz june_21/real/NW_ppo_nofb.npz -tt 4.2 --hovertime 4.0 --runtime 12.0 -bh 0.6
ffmpeg -framerate 20 -i imgs/%04d.png -pix_fmt yuv420p videos/gui_NW_ppo_pid_mppi.mp4 -y