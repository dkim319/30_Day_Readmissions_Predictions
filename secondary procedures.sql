select distinct p.hadm_id
	, case when icd_p.long_title is not null
	then icd_p.long_title
	else p.icd9_code end long_title
from mimiciii.procedures_icd p
left join mimiciii.d_icd_procedures icd_p
	on icd_p.icd9_code = p.icd9_code
	--on trim(leading '0' from icd_p.icd9_code) = trim(leading '0' from p.icd9_code)
where p.seq_num > 1;
