# parameter - prefix to sensemap files
for pos in noun verb
do
	awk '{print $2"\t"$4}' ${1}.${pos}.mono |sort | uniq > ${1}-${pos}-mono-mapping.tsv
	awk -F" |;" '{if (NF > 5) print $3"\t"$6}' ${1}.${pos}.poly | sort | uniq > ${1}-${pos}-poly-mapping.tsv
	awk -F" |;" '{if (NF > 8) print $3"\t"$9}' ${1}.${pos}.poly | sort | uniq >> ${1}-${pos}-poly-mapping.tsv
	awk -F" |;" '{if (NF > 11) print $3"\t"$12}' ${1}.${pos}.poly | sort | uniq >> ${1}-${pos}-poly-mapping.tsv
	cat ${1}-${pos}-*-mapping.tsv | sort | uniq > ${1}-${pos}-all-mapping.tsv
done
