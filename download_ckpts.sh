mkdir -p Grounded-SAM-2/checkpoints
cd Grounded-SAM-2/checkpoints
echo "Downloading SAM 2 Hiera Large..."
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
echo "Downloading Grounding DINO SwinT OGC..."
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
echo "Done! Checkpoints are ready."
