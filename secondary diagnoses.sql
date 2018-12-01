select distinct d.hadm_id
	, case when icd_d.long_title is not null
	then icd_d.long_title
	else d.icd9_code end long_title
from mimiciii.diagnoses_icd d
left join mimiciii.d_icd_diagnoses icd_d
	on icd_d.icd9_code = d.icd9_code
	--on trim(leading '0' from icd_d.icd9_code) = trim(leading '0' from d.icd9_code)
where d.seq_num > 1;