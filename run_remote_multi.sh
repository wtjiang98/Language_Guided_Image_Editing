filename=`basename "$1"`
# encode the image name
filenamebase64=$(echo $filename | base64)
filenamebase64=$(echo ${filenamebase64:0:8})
demoindir='/u/jshi31/project/LDIE/output/FiveK_trial_1/demo_input'
demooutdir='/u/jshi31/project/LDIE/output/FiveK_trial_1/demo_output'
input_path=$demoindir/$filenamebase64
output_path=$demooutdir/$filenamebase64
# upload the image
scp $1 jshi31@cornea.cs.rochester.edu:$input_path
# call model
ssh jshi31@cornea.cs.rochester.edu /u/jshi31/project/LDIE/run_demo_FiveK.sh $input_path 1 $2
# download the image
scp -r jshi31@cornea.cs.rochester.edu:${output_path}/* $3
