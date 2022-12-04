datasets=("2d-20c-no0.arff" "complex9.arff" " st900-2-9.arff" "hypercube.arff" "chainlink.arff" "bio-protein.arff" "hepta.arff")
alphas=("2.5" "0.124" "6.565" "4.375" "2.5" "0.0051" "1.249")

n="${#datasets[@]}"
for i in {0..6}; do
    echo "----------------------------------------------------------------------"
    echo "Running dataset ${datasets[i]} with alpha ${alphas[i]}"
    echo "----------------------------------------------------------------------"
    python3 main.py ${datasets[i]} ${alphas[i]}
done