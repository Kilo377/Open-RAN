"""Optional scheduled-link measurements cannot change the CSV header late."""
import csv
import io

from experiments.run_rt_sys import ue_csv_record


def test_unscheduled_first_row_keeps_later_effective_sinr_without_fabricating_zero():
    idle=dict(ue_id='ue_0',scheduled=False,sinr_db=0.)
    scheduled=dict(ue_id='ue_1',scheduled=True,sinr_db=9.,effective_sinr_db=8.25)
    original=dict(idle);stream=io.StringIO();first=ue_csv_record(40,idle)
    writer=csv.DictWriter(stream,fieldnames=list(first));writer.writeheader()
    writer.writerow(first);writer.writerow(ue_csv_record(40,scheduled))
    rows=list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert rows[0]['effective_sinr_db']==''
    assert float(rows[1]['effective_sinr_db'])==8.25
    assert idle==original and 'effective_sinr_db'not in idle


def test_entire_first_slot_may_be_unscheduled_before_the_first_transmission():
    idle=dict(ue_id='ue_0',scheduled=False,sinr_db=0.)
    stream=io.StringIO();first=ue_csv_record(40,idle);writer=csv.DictWriter(stream,fieldnames=list(first));writer.writeheader()
    writer.writerow(first);writer.writerow(ue_csv_record(80,dict(idle,scheduled=True,effective_sinr_db=-2.5)))
    rows=list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert rows[0]['effective_sinr_db']=='' and rows[1]['effective_sinr_db']=='-2.5'
