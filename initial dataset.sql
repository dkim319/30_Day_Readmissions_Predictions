with icu as (select hadm_id
			, sum(los) icu_los
			from mimiciii.icustays icu
			group by hadm_id)
, drgcodes as (select distinct d.hadm_id
	, first_value(d.row_id) over (partition by d.hadm_id order by d.row_id) row_id
from mimiciii.drgcodes d)
, drgcodes_clean as (select d.*
	, d2.description
from drgcodes d
inner join mimiciii.drgcodes d2
	on d2.row_id = d.row_id)
, primary_cancer as (select d.hadm_id
	, d.seq_num
	, icd_d.long_title
	, cast(substr(d.icd9_code, 3) as integer)
from mimiciii.diagnoses_icd d
left join mimiciii.d_icd_diagnoses icd_d
	on icd_d.icd9_code = d.icd9_code
where (icd_d.row_id between 914 and 956
or icd_d.row_id between 1141 and 1350
or icd_d.row_id between 1524 and 1590
or icd_d.row_id between 1681 and 1716
or icd_d.row_id between 1835 and 2152
or icd_d.row_id between 2317 and 2395)
and d.seq_num = 1)
select distinct a.subject_id
	, a.hadm_id
	, a.admittime
	, a.dischtime
	, a.admission_type
	, a.admission_location
	, a.discharge_location
	, a.insurance
	, a.language
	, a.religion
	, a.marital_status
	, a.ethnicity
	, a.diagnosis
	, drg.description drg
	, extract(day from a.dischtime -  a.admittime) + extract(hour from a.dischtime -  a.admittime) / 24 as los_days
	, extract(minute from a.edouttime - a.edregtime) / 60 as emergency_hours
	, a.admittime
	, p.dob
	, floor(extract(day from a.admittime - p.dob) / 365) as age
	, i.icu_los icu_days
	, pp.long_title primary_proc
	, pd.long_title primary_diag
	, dc.diag_count
	, pc.proc_count
	, (select count(*)
	  from mimiciii.admissions a2
	  where a2.subject_id = a.subject_id
	  	and a2.dischtime < a.admittime
	    and extract(day from a2.dischtime - a.admittime) <= 30) visits_prior_30_days
	, (select count(*)
	  from mimiciii.admissions a2
	  where a2.subject_id = a.subject_id
	  	and a2.dischtime < a.admittime
	    and extract(day from a2.dischtime - a.admittime) <= 30
	  	and a2.admission_type <> 'ELECTIVE') non_elective_visits_prior_30_days
	, (select count(*)
	  from mimiciii.admissions a2
	  where a2.subject_id = a.subject_id
	  	and a2.dischtime < a.admittime
	    and extract(day from a2.dischtime - a.admittime) <= 365) visits_prior_year
	, (select count(*)
	  from mimiciii.admissions a2
	  where a2.subject_id = a.subject_id
	  	and a2.dischtime < a.admittime
	    and extract(day from a2.dischtime - a.admittime) <= 365
	  	and a2.admission_type <> 'ELECTIVE') nonelective_visits_prior_year
	, case when cancer.hadm_id is not null
	then 1
	else 0 end cancer_flag
	, case when (select count(a2.hadm_id)
		from mimiciii.admissions a2
		where a2.subject_id = a.subject_id
		and a2.admission_type <> 'ELECTIVE'
		and extract(day from a2.admittime - a.dischtime) <= 30
		and a2.admittime > a.dischtime
		and a2.hadm_id <> a.hadm_id) > 0
	then 1
	else 0 end readmitted_flag 
	, a.hospital_expire_flag
from mimiciii.admissions a
left join mimiciii.patients p
	on p.subject_id = a.subject_id
left join drgcodes_clean drg
	on drg.hadm_id = a.hadm_id
	--and drg.drg_type = 'APR'
left join icu i
	on i.hadm_id = a.hadm_id
left join (select p.hadm_id
	, case when icd_p.long_title is not null
	then icd_p.long_title
	else p.icd9_code end long_title
from mimiciii.procedures_icd p
left join mimiciii.d_icd_procedures icd_p
	on icd_p.icd9_code = p.icd9_code
	--on trim(leading '0' from icd_p.icd9_code) = trim(leading '0' from p.icd9_code)
where p.seq_num = 1) pp
	on pp.hadm_id = a.hadm_id
left join (select d.hadm_id
	, case when icd_d.long_title is not null
	then icd_d.long_title
	else d.icd9_code end long_title
from mimiciii.diagnoses_icd d
left join mimiciii.d_icd_diagnoses icd_d
	on icd_d.icd9_code = d.icd9_code
	--on trim(leading '0' from icd_d.icd9_code) = trim(leading '0' from d.icd9_code)
where d.seq_num = 1) pd
	on pd.hadm_id = a.hadm_id
left join (select hadm_id
		  , count(seq_num) diag_count
	from mimiciii.diagnoses_icd
	group by hadm_id) dc
	on dc.hadm_id = a.hadm_id
left join (select hadm_id
		  , count(seq_num) proc_count
	from mimiciii.procedures_icd
	group by hadm_id) pc
	on pc.hadm_id = a.hadm_id
left join primary_cancer cancer
	on cancer.hadm_id = a.hadm_id
where a.diagnosis <> 'ORGAN DONOR ACCOUNT';