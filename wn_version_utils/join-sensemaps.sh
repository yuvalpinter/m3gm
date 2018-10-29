for pos in verb noun
do
	sort -k2 1.7.1to2.0-${pos}-all-mapping.tsv > 1.7.1to2.0-${pos}-all-mapping-s.tsv
	join 1.7.1to2.0-${pos}-all-mapping-s.tsv 2.0to2.1-${pos}-all-mapping.tsv -1 2 | awk '{print $2"\t"$3}' | sort -k2 > 1.7.1to2.1-${pos}-all-mapping-s.tsv
	join 1.7.1to2.1-${pos}-all-mapping-s.tsv 2.1to3.0-${pos}-all-mapping.tsv -1 2 | awk '{print $2"\t"$3}' | sort > 1.7.1to3.0-${pos}-all-mapping.tsv
done
