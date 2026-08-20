from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import pm4py  # type: ignore
except Exception:
    pm4py = None


RECOMMENDED_SPLIT = {
    "pretrain": [
        "BPIC2013I",
        "BPIC2013O",
        "BPIC2019",
        "Road_Traffic_Fine",
        "BPIC2020D",
        "BPIC2020Pe",
        "BPIC2020Pr",
        "BPIC2020I",
        "BPIC2012A",
        "BPIC2012O",
        "Receipt",
        "Sepsis",
        "Service-process",
    ],
    "test": [
        "BPIC2012W",
        "BPIC2020R",
        "BPIC2017O",
        "BPIC2013C",
        "Hospital-billing",
        "Helpdesk",
    ],
}

ACTIVITY_MAPPINGS: Dict[str, Dict[str, str]] = {'BPIC2013I': {'Accepted': 'accept incident',
               'Completed': 'complete incident',
               'Queued': 'queue incident',
               'Unmatched': 'flag unmatched incident'},
 'BPIC2013O': {'Accepted': 'accept problem', 'Completed': 'complete problem', 'Queued': 'queue problem'},
 'BPIC2013C': {'Accepted': 'accept problem',
               'Completed': 'complete problem',
               'Queued': 'queue problem',
               'Unmatched': 'flag unmatched problem'},
 'BPIC2017O': {'O_Accepted': 'accept offer',
               'O_Cancelled': 'cancel offer',
               'O_Create Offer': 'create offer',
               'O_Created': 'register offer',
               'O_Refused': 'refuse offer',
               'O_Returned': 'return offer',
               'O_Sent (mail and online)': 'send offer by mail and online',
               'O_Sent (online only)': 'send offer online'},
 'BPIC2019': {'Block Purchase Order Item': 'block purchase order item',
              'Cancel Goods Receipt': 'cancel goods receipt',
              'Cancel Invoice Receipt': 'cancel invoice receipt',
              'Cancel Subsequent Invoice': 'cancel subsequent invoice',
              'Change Approval for Purchase Order': 'change purchase order approval',
              'Change Currency': 'change currency',
              'Change Delivery Indicator': 'change delivery indicator',
              'Change Final Invoice Indicator': 'change final invoice indicator',
              'Change payment term': 'change payment term',
              'Change Price': 'change price',
              'Change Quantity': 'change quantity',
              'Change Rejection Indicator': 'change rejection indicator',
              'Change Storage Location': 'change storage location',
              'Clear Invoice': 'clear invoice',
              'Create Purchase Order Item': 'create purchase order item',
              'Create Purchase Requisition Item': 'create purchase requisition item',
              'Delete Purchase Order Item': 'delete purchase order item',
              'Reactivate Purchase Order Item': 'reactivate purchase order item',
              'Receive Order Confirmation': 'receive order confirmation',
              'Record Goods Receipt': 'record goods receipt',
              'Record Invoice Receipt': 'record invoice receipt',
              'Record Service Entry Sheet': 'record service entry sheet',
              'Record Subsequent Invoice': 'record subsequent invoice',
              'Release Purchase Order': 'release purchase order',
              'Release Purchase Requisition': 'release purchase requisition',
              'Remove Payment Block': 'remove payment block',
              'Set Payment Block': 'set payment block',
              'SRM: Awaiting Approval': 'await srm approval',
              'SRM: Change was Transmitted': 'transmit srm change',
              'SRM: Complete': 'complete srm record',
              'SRM: Created': 'create srm document',
              'SRM: Deleted': 'delete srm document',
              'SRM: Document Completed': 'complete srm document',
              'SRM: Held': 'hold srm document',
              'SRM: In Transfer to Execution Syst.': 'transfer srm to execution system',
              'SRM: Incomplete': 'mark srm incomplete',
              'SRM: Ordered': 'order in srm',
              'SRM: Transaction Completed': 'complete srm transaction',
              'SRM: Transfer Failed (E.Sys.)': 'fail srm transfer to execution system',
              'Update Order Confirmation': 'update order confirmation',
              'Vendor creates debit memo': 'create vendor debit memo',
              'Vendor creates invoice': 'create vendor invoice'},
 'BPIC2012A': {'ACCEPTED': 'accept application',
               'ACTIVATED': 'activate application',
               'APPROVED': 'approve application',
               'CANCELLED': 'cancel application',
               'DECLINED': 'decline application',
               'FINALIZED': 'finalize application',
               'PARTLYSUBMITTED': 'partially submit application',
               'PREACCEPTED': 'preaccept application',
               'REGISTERED': 'register application',
               'SUBMITTED': 'submit application'},
 'BPIC2012O': {'ACCEPTED': 'accept offer',
               'CANCELLED': 'cancel offer',
               'CREATED': 'create offer',
               'DECLINED': 'decline offer',
               'SELECTED': 'select offer',
               'SENT': 'send offer',
               'SENT_BACK': 'return offer'},
 'BPIC2012W': {'Afhandelen leads': 'follow up lead',
               'Beoordelen fraude': 'assess fraud',
               'Completeren aanvraag': 'complete application',
               'Nabellen incomplete dossiers': 'follow up incomplete dossier',
               'Nabellen offertes': 'follow up offer',
               'Valideren aanvraag': 'validate application'},
 'Hospital-billing': {'BILLED': 'issue bill',
                      'CHANGE DIAGN': 'change diagnosis code',
                      'CHANGE END': 'change end date',
                      'CODE ERROR': 'flag coding error',
                      'CODE NOK': 'mark coding not ok',
                      'CODE OK': 'mark coding ok',
                      'DELETE': 'delete billing record',
                      'EMPTY': 'mark bill as empty',
                      'FIN': 'finalize bill',
                      'JOIN-PAT': 'link patient record',
                      'MANUAL': 'perform manual billing adjustment',
                      'NEW': 'create bill',
                      'REJECT': 'reject bill',
                      'RELEASE': 'release bill',
                      'REOPEN': 'reopen bill',
                      'SET STATUS': 'set billing status',
                      'STORNO': 'reverse bill',
                      'ZDBC_BEHAN': 'process treatment record'},
 'Receipt': {'Confirmation of receipt': 'confirm receipt',
             'T02 Check confirmation of receipt': 'check receipt confirmation',
             'T03 Adjust confirmation of receipt': 'adjust receipt confirmation',
             'T04 Determine confirmation of receipt': 'determine receipt confirmation',
             'T05 Print and send confirmation of receipt': 'send receipt confirmation',
             'T06 Determine necessity of stop advice': 'determine stop advice need',
             'T07-1 Draft intern advice aspect 1': 'draft internal advice aspect 1',
             'T07-2 Draft intern advice aspect 2': 'draft internal advice aspect 2',
             'T07-3 Draft intern advice hold for aspect 3': 'draft internal advice aspect 3',
             'T07-4 Draft internal advice to hold for type 4': 'draft internal advice aspect 4',
             'T07-5 Draft intern advice aspect 5': 'draft internal advice aspect 5',
             'T08 Draft and send request for advice': 'send advice request',
             'T09-1 Process or receive external advice from party 1': 'process external advice party 1',
             'T09-2 Process or receive external advice from party 2': 'process external advice party 2',
             'T09-3 Process or receive external advice from party 3': 'process external advice party 3',
             'T09-4 Process or receive external advice from party 4': 'process external advice party 4',
             'T10 Determine necessity to stop indication': 'determine stop indication need',
             'T11 Create document X request unlicensed': 'create unlicensed request document',
             'T12 Check document X request unlicensed': 'check unlicensed request document',
             'T13 Adjust document X request unlicensed': 'adjust unlicensed request document',
             'T14 Determine document X request unlicensed': 'determine unlicensed request document',
             'T15 Print document X request unlicensed': 'print unlicensed request document',
             'T16 Report reasons to hold request': 'report hold reasons',
             'T17 Check report Y to stop indication': 'check stop indication report',
             'T18 Adjust report Y to stop indicition': 'adjust stop indication report',
             'T19 Determine report Y to stop indication': 'determine stop indication report',
             'T20 Print report Y to stop indication': 'print stop indication report'},
 'Sepsis': {'Admission IC': 'admit patient to intensive care',
            'Admission NC': 'admit patient to normal care',
            'CRP': 'test crp',
            'ER Registration': 'register emergency visit',
            'ER Sepsis Triage': 'triage sepsis in emergency',
            'ER Triage': 'triage patient in emergency',
            'IV Antibiotics': 'administer iv antibiotics',
            'IV Liquid': 'administer iv fluid',
            'LacticAcid': 'test lactic acid',
            'Leucocytes': 'test leucocytes',
            'Release A': 'release patient a',
            'Release B': 'release patient b',
            'Release C': 'release patient c',
            'Release D': 'release patient d',
            'Release E': 'release patient e',
            'Return ER': 'return patient to emergency'},
 'Helpdesk': {'Assign seriousness': 'assign severity',
              'Closed': 'close ticket',
              'Create SW anomaly': 'create software anomaly',
              'DUPLICATE': 'mark duplicate ticket',
              'Insert ticket': 'record ticket',
              'INVALID': 'mark invalid ticket',
              'Require upgrade': 'request upgrade',
              'Resolve SW anomaly': 'resolve software anomaly',
              'Resolve ticket': 'resolve ticket',
              'RESOLVED': 'mark ticket resolved',
              'Schedule intervention': 'schedule intervention',
              'Take in charge ticket': 'take ownership of ticket',
              'VERIFIED': 'verify ticket',
              'Wait': 'wait for action'},
 'Road_Traffic_Fine': {'Add penalty': 'add penalty',
                       'Appeal to Judge': 'submit judge appeal',
                       'Create Fine': 'create fine',
                       'Insert Date Appeal to Prefecture': 'record prefecture appeal date',
                       'Insert Fine Notification': 'record fine notification',
                       'Notify Result Appeal to Offender': 'notify appeal result',
                       'Payment': 'receive payment',
                       'Receive Result Appeal from Prefecture': 'receive prefecture appeal result',
                       'Send Appeal to Prefecture': 'send appeal to prefecture',
                       'Send Fine': 'send fine',
                       'Send for Credit Collection': 'send case for debt collection'},
 'Service-process': {'Approved': 'approve service order',
                     'Completed': 'complete service order',
                     'Creation': 'create service order',
                     'DeviceReceived': 'receive device',
                     'FreeticketComp': 'create free company ticket',
                     'FreeticketCust': 'create free customer ticket',
                     'InDelivery': 'deliver device',
                     'Letter': 'send letter',
                     'NoteHotline': 'record hotline note',
                     'NoteWorkshop': 'record workshop note',
                     'StatusRequest': 'request status',
                     'StockEntry': 'record stock entry',
                     'Transmission': 'transmit service order'},
 'Production': {'Change Version - Machine 22': 'change version',
                'Deburring - Manual': 'deburr part',
                'Final Inspection - Weighting': 'weigh part',
                'Final Inspection Q.C.': 'inspect part',
                'Fix - Machine 15': 'fix part on machine 15',
                'Fix - Machine 15M': 'fix part on machine 15m',
                'Fix - Machine 19': 'fix part on machine 19',
                'Fix - Machine 3': 'fix part on machine 3',
                'Fix EDM': 'fix edm process',
                'Flat Grinding - Machine 11': 'flat grind part on machine 11',
                'Flat Grinding - Machine 26': 'flat grind part on machine 26',
                'Grinding Rework': 'rework grinding',
                'Grinding Rework - Machine 12': 'rework grinding on machine 12',
                'Grinding Rework - Machine 2': 'rework grinding on machine 2',
                'Grinding Rework - Machine 27': 'rework grinding on machine 27',
                'Lapping - Machine 1': 'lap part',
                'Laser Marking - Machine 7': 'laser mark part',
                'Milling - Machine 10': 'mill part on machine 10',
                'Milling - Machine 14': 'mill part on machine 14',
                'Milling - Machine 16': 'mill part on machine 16',
                'Milling - Machine 8': 'mill part on machine 8',
                'Milling Q.C.': 'inspect milling',
                'Nitration Q.C.': 'inspect nitration',
                'Packing': 'pack part',
                'Rework Milling - Machine 28': 'rework milling',
                'Round  Q.C.': 'inspect round part',
                'Round Grinding - Machine 12': 'round grind part on machine 12',
                'Round Grinding - Machine 19': 'round grind part on machine 19',
                'Round Grinding - Machine 2': 'round grind part on machine 2',
                'Round Grinding - Machine 23': 'round grind part on machine 23',
                'Round Grinding - Machine 3': 'round grind part on machine 3',
                'Round Grinding - Manual': 'round grind part manually',
                'Round Grinding - Q.C.': 'inspect round grinding',
                'SETUP     Turning & Milling - Machine 5': 'setup turning and milling',
                'Setup - Machine 4': 'setup machine on machine 4',
                'Setup - Machine 8': 'setup machine on machine 8',
                'Stress Relief': 'stress relieve part',
                'Turn & Mill. & Screw Assem - Machine 10': 'turn mill and screw assemble part on machine 10',
                'Turn & Mill. & Screw Assem - Machine 9': 'turn mill and screw assemble part on machine 9',
                'Turning & Milling - Machine 10': 'turn and mill part on machine 10',
                'Turning & Milling - Machine 4': 'turn and mill part on machine 4',
                'Turning & Milling - Machine 5': 'turn and mill part on machine 5',
                'Turning & Milling - Machine 6': 'turn and mill part on machine 6',
                'Turning & Milling - Machine 8': 'turn and mill part on machine 8',
                'Turning & Milling - Machine 9': 'turn and mill part on machine 9',
                'Turning & Milling Q.C.': 'inspect turning and milling',
                'Turning - Machine 21': 'turn part on machine 21',
                'Turning - Machine 4': 'turn part on machine 4',
                'Turning - Machine 5': 'turn part on machine 5',
                'Turning - Machine 8': 'turn part on machine 8',
                'Turning - Machine 9': 'turn part on machine 9',
                'Turning Q.C.': 'inspect turning',
                'Turning Rework - Machine 21': 'rework turning',
                'Wire Cut - Machine 13': 'wire cut part on machine 13',
                'Wire Cut - Machine 18': 'wire cut part on machine 18'},
 'BPIC2020D': {'Declaration APPROVED by ADMINISTRATION': 'approve declaration by administration',
               'Declaration APPROVED by BUDGET OWNER': 'approve declaration by budget owner',
               'Declaration APPROVED by PRE_APPROVER': 'approve declaration by pre approver',
               'Declaration FINAL_APPROVED by SUPERVISOR': 'final approve declaration',
               'Declaration FOR_APPROVAL by ADMINISTRATION': 'forward declaration for approval by administration',
               'Declaration FOR_APPROVAL by PRE_APPROVER': 'forward declaration for approval by pre approver',
               'Declaration FOR_APPROVAL by SUPERVISOR': 'forward declaration for approval by supervisor',
               'Declaration REJECTED by ADMINISTRATION': 'reject declaration by administration',
               'Declaration REJECTED by BUDGET OWNER': 'reject declaration by budget owner',
               'Declaration REJECTED by EMPLOYEE': 'reject declaration by employee',
               'Declaration REJECTED by MISSING': 'reject declaration by missing',
               'Declaration REJECTED by PRE_APPROVER': 'reject declaration by pre approver',
               'Declaration REJECTED by SUPERVISOR': 'reject declaration by supervisor',
               'Declaration SAVED by EMPLOYEE': 'save declaration',
               'Declaration SUBMITTED by EMPLOYEE': 'submit declaration',
               'Payment Handled': 'handle payment',
               'Request Payment': 'request payment'},
 'BPIC2020I': {'Declaration APPROVED by ADMINISTRATION': 'approve declaration by administration',
               'Declaration APPROVED by BUDGET OWNER': 'approve declaration by budget owner',
               'Declaration APPROVED by PRE_APPROVER': 'approve declaration by pre approver',
               'Declaration APPROVED by SUPERVISOR': 'approve declaration by supervisor',
               'Declaration FINAL_APPROVED by DIRECTOR': 'final approve declaration by director',
               'Declaration FINAL_APPROVED by SUPERVISOR': 'final approve declaration by supervisor',
               'Declaration REJECTED by ADMINISTRATION': 'reject declaration by administration',
               'Declaration REJECTED by BUDGET OWNER': 'reject declaration by budget owner',
               'Declaration REJECTED by DIRECTOR': 'reject declaration by director',
               'Declaration REJECTED by EMPLOYEE': 'reject declaration by employee',
               'Declaration REJECTED by MISSING': 'reject declaration by missing',
               'Declaration REJECTED by PRE_APPROVER': 'reject declaration by pre approver',
               'Declaration REJECTED by SUPERVISOR': 'reject declaration by supervisor',
               'Declaration SAVED by EMPLOYEE': 'save declaration',
               'Declaration SUBMITTED by EMPLOYEE': 'submit declaration',
               'End trip': 'end trip',
               'Payment Handled': 'handle payment',
               'Permit APPROVED by ADMINISTRATION': 'approve permit by administration',
               'Permit APPROVED by BUDGET OWNER': 'approve permit by budget owner',
               'Permit APPROVED by PRE_APPROVER': 'approve permit by pre approver',
               'Permit APPROVED by SUPERVISOR': 'approve permit by supervisor',
               'Permit FINAL_APPROVED by DIRECTOR': 'final approve permit by director',
               'Permit FINAL_APPROVED by SUPERVISOR': 'final approve permit by supervisor',
               'Permit REJECTED by ADMINISTRATION': 'reject permit by administration',
               'Permit REJECTED by BUDGET OWNER': 'reject permit by budget owner',
               'Permit REJECTED by DIRECTOR': 'reject permit by director',
               'Permit REJECTED by EMPLOYEE': 'reject permit by employee',
               'Permit REJECTED by MISSING': 'reject permit by missing',
               'Permit REJECTED by PRE_APPROVER': 'reject permit by pre approver',
               'Permit REJECTED by SUPERVISOR': 'reject permit by supervisor',
               'Permit SUBMITTED by EMPLOYEE': 'submit permit',
               'Request Payment': 'request payment',
               'Send Reminder': 'send reminder',
               'Start trip': 'start trip'},
 'BPIC2020Pe': {'Payment Handled': 'handle payment',
                'Permit APPROVED by ADMINISTRATION': 'approve permit by administration',
                'Permit APPROVED by BUDGET OWNER': 'approve permit by budget owner',
                'Permit APPROVED by PRE_APPROVER': 'approve permit by pre approver',
                'Permit APPROVED by SUPERVISOR': 'approve permit by supervisor',
                'Permit FINAL_APPROVED by DIRECTOR': 'final approve permit by director',
                'Permit FINAL_APPROVED by SUPERVISOR': 'final approve permit by supervisor',
                'Permit REJECTED by ADMINISTRATION': 'reject permit by administration',
                'Permit REJECTED by BUDGET OWNER': 'reject permit by budget owner',
                'Permit REJECTED by EMPLOYEE': 'reject permit by employee',
                'Permit REJECTED by MISSING': 'reject permit by missing',
                'Permit REJECTED by PRE_APPROVER': 'reject permit by pre approver',
                'Permit REJECTED by SUPERVISOR': 'reject permit by supervisor',
                'Permit SUBMITTED by EMPLOYEE': 'submit permit',
                'Request For Payment APPROVED by ADMINISTRATION': 'approve payment request by administration',
                'Request For Payment APPROVED by BUDGET OWNER': 'approve payment request by budget owner',
                'Request For Payment APPROVED by PRE_APPROVER': 'approve payment request by pre approver',
                'Request For Payment APPROVED by SUPERVISOR': 'approve payment request by supervisor',
                'Request For Payment FINAL_APPROVED by DIRECTOR': 'final approve payment request by director',
                'Request For Payment FINAL_APPROVED by SUPERVISOR': 'final approve payment request by supervisor',
                'Request For Payment REJECTED by ADMINISTRATION': 'reject payment request by administration',
                'Request For Payment REJECTED by BUDGET OWNER': 'reject payment request by budget owner',
                'Request For Payment REJECTED by EMPLOYEE': 'reject payment request by employee',
                'Request For Payment REJECTED by MISSING': 'reject payment request by missing',
                'Request For Payment REJECTED by PRE_APPROVER': 'reject payment request by pre approver',
                'Request For Payment REJECTED by SUPERVISOR': 'reject payment request by supervisor',
                'Request For Payment SAVED by EMPLOYEE': 'save payment request',
                'Request For Payment SUBMITTED by EMPLOYEE': 'submit payment request',
                'Request Payment': 'request payment',
                'Declaration APPROVED by ADMINISTRATION': 'approve declaration by administration',
                'Declaration APPROVED by BUDGET OWNER': 'approve declaration by budget owner',
                'Declaration APPROVED by PRE_APPROVER': 'approve declaration by pre approver',
                'Declaration APPROVED by SUPERVISOR': 'approve declaration by supervisor',
                'Declaration FINAL_APPROVED by DIRECTOR': 'final approve declaration by director',
                'Declaration FINAL_APPROVED by SUPERVISOR': 'final approve declaration by supervisor',
                'Declaration REJECTED by ADMINISTRATION': 'reject declaration by administration',
                'Declaration REJECTED by BUDGET OWNER': 'reject declaration by budget owner',
                'Declaration REJECTED by DIRECTOR': 'reject declaration by director',
                'Declaration REJECTED by EMPLOYEE': 'reject declaration by employee',
                'Declaration REJECTED by MISSING': 'reject declaration by missing',
                'Declaration REJECTED by PRE_APPROVER': 'reject declaration by pre approver',
                'Declaration REJECTED by SUPERVISOR': 'reject declaration by supervisor',
                'Declaration SAVED by EMPLOYEE': 'save declaration',
                'Declaration SUBMITTED by EMPLOYEE': 'submit declaration',
                'End trip': 'end trip',
                'Permit FOR_APPROVAL by ADMINISTRATION': 'forward permit for approval by administration',
                'Permit FOR_APPROVAL by SUPERVISOR': 'forward permit for approval by supervisor',
                'Permit REJECTED by DIRECTOR': 'reject permit by director',
                'Permit SAVED by EMPLOYEE': 'save permit',
                'Send Reminder': 'send reminder',
                'Start trip': 'start trip'},
 'BPIC2020Pr': {'Payment Handled': 'handle payment',
                'Permit APPROVED by ADMINISTRATION': 'approve permit by administration',
                'Permit APPROVED by BUDGET OWNER': 'approve permit by budget owner',
                'Permit APPROVED by PRE_APPROVER': 'approve permit by pre approver',
                'Permit APPROVED by SUPERVISOR': 'approve permit by supervisor',
                'Permit FINAL_APPROVED by DIRECTOR': 'final approve permit by director',
                'Permit FINAL_APPROVED by SUPERVISOR': 'final approve permit by supervisor',
                'Permit REJECTED by ADMINISTRATION': 'reject permit by administration',
                'Permit REJECTED by BUDGET OWNER': 'reject permit by budget owner',
                'Permit REJECTED by EMPLOYEE': 'reject permit by employee',
                'Permit REJECTED by MISSING': 'reject permit by missing',
                'Permit REJECTED by PRE_APPROVER': 'reject permit by pre approver',
                'Permit REJECTED by SUPERVISOR': 'reject permit by supervisor',
                'Permit SUBMITTED by EMPLOYEE': 'submit permit',
                'Request For Payment APPROVED by ADMINISTRATION': 'approve payment request by administration',
                'Request For Payment APPROVED by BUDGET OWNER': 'approve payment request by budget owner',
                'Request For Payment APPROVED by PRE_APPROVER': 'approve payment request by pre approver',
                'Request For Payment APPROVED by SUPERVISOR': 'approve payment request by supervisor',
                'Request For Payment FINAL_APPROVED by DIRECTOR': 'final approve payment request by director',
                'Request For Payment FINAL_APPROVED by SUPERVISOR': 'final approve payment request by supervisor',
                'Request For Payment REJECTED by ADMINISTRATION': 'reject payment request by administration',
                'Request For Payment REJECTED by BUDGET OWNER': 'reject payment request by budget owner',
                'Request For Payment REJECTED by EMPLOYEE': 'reject payment request by employee',
                'Request For Payment REJECTED by MISSING': 'reject payment request by missing',
                'Request For Payment REJECTED by PRE_APPROVER': 'reject payment request by pre approver',
                'Request For Payment REJECTED by SUPERVISOR': 'reject payment request by supervisor',
                'Request For Payment SAVED by EMPLOYEE': 'save payment request',
                'Request For Payment SUBMITTED by EMPLOYEE': 'submit payment request',
                'Request Payment': 'request payment'},
 'BPIC2020R': {'Payment Handled': 'handle payment',
               'Request For Payment APPROVED by ADMINISTRATION': 'approve payment request by administration',
               'Request For Payment APPROVED by BUDGET OWNER': 'approve payment request by budget owner',
               'Request For Payment APPROVED by PRE_APPROVER': 'approve payment request by pre approver',
               'Request For Payment APPROVED by SUPERVISOR': 'approve payment request by supervisor',
               'Request For Payment FINAL_APPROVED by BUDGET OWNER': 'final approve payment request by budget owner',
               'Request For Payment FINAL_APPROVED by DIRECTOR': 'final approve payment request by director',
               'Request For Payment FINAL_APPROVED by SUPERVISOR': 'final approve payment request by supervisor',
               'Request For Payment FOR_APPROVAL by ADMINISTRATION': 'forward payment request for approval by administration',
               'Request For Payment FOR_APPROVAL by SUPERVISOR': 'forward payment request for approval by supervisor',
               'Request For Payment REJECTED by ADMINISTRATION': 'reject payment request by administration',
               'Request For Payment REJECTED by BUDGET OWNER': 'reject payment request by budget owner',
               'Request For Payment REJECTED by EMPLOYEE': 'reject payment request by employee',
               'Request For Payment REJECTED by MISSING': 'reject payment request by missing',
               'Request For Payment REJECTED by PRE_APPROVER': 'reject payment request by pre approver',
               'Request For Payment REJECTED by SUPERVISOR': 'reject payment request by supervisor',
               'Request For Payment SAVED by EMPLOYEE': 'save payment request',
               'Request For Payment SUBMITTED by EMPLOYEE': 'submit payment request',
               'Request Payment': 'request payment'}}
DATASET_ALIASES: Dict[str, str] = {'BPI_Challenge_2013_I': 'BPIC2013I',
 'BPI_Challenge_2013_O': 'BPIC2013O',
 'BPI_Challenge_2013_C': 'BPIC2013C',
 'BPI_Challenge_2017_O': 'BPIC2017O',
 'BPI_Challenge_2019': 'BPIC2019',
 'BPI_Challenge_2020_D': 'BPIC2020D',
 'BPI_Challenge_2020_I': 'BPIC2020I',
 'BPI_Challenge_2020_Pe': 'BPIC2020Pe',
 'BPI_Challenge_2020_Pr': 'BPIC2020Pr',
 'BPI_Challenge_2020_R': 'BPIC2020R',
 'BPIC_2012_A': 'BPIC2012A',
 'BPIC_2012_O': 'BPIC2012O',
 'BPIC2012-O': 'BPIC2012O',
 'BPIC_2012_W': 'BPIC2012W',
 'BPIC2012-W': 'BPIC2012W',
 'HOSPITAL-BILLING': 'Hospital-billing',
 'HOSPITAL_BILLING': 'Hospital-billing',
 'HOSPITAL BILLING': 'Hospital-billing',
 'RECEIPT': 'Receipt',
 'SEPSIS': 'Sepsis',
 'HELPDESK': 'Helpdesk',
 'PRODUCTION': 'Production',
 'ROAD_TRAFFIC_FINE': 'Road_Traffic_Fine',
 'ROAD TRAFFIC FINE': 'Road_Traffic_Fine',
 'ROAD-TRAFFIC-FINE': 'Road_Traffic_Fine',
 'SERVICE-PROCESS': 'Service-process',
 'SERVICE_PROCESS': 'Service-process',
 'Service_process': 'Service-process'}
DATASET_PATH_HINTS: Dict[str, List[str]] = {
    "BPIC2013I": ["bpic2013i", "bpi2013i", "incidentmanagement", "incidents"],
    "BPIC2013O": ["bpic2013o", "bpi2013o", "openproblems"],
    "BPIC2013C": ["bpic2013c", "bpi2013c", "closedproblems"],
    "BPIC2017O": ["bpic2017o", "bpi2017o"],
    "BPIC2019": ["bpic2019", "bpi2019"],
    "BPIC2020D": ["bpic2020d", "domesticdeclarations", "domesticdeclaration"],
    "BPIC2020I": ["bpic2020i", "internationaldeclarations", "internationaldeclaration"],
    "BPIC2020Pe": ["bpic2020pe", "travelpermits", "travelpermit"],
    "BPIC2020Pr": ["bpic2020pr", "prepaidtravelcosts", "prepaidtravelcost"],
    "BPIC2020R": ["bpic2020r", "requestforpayment", "requestsforpayment"],
    "BPIC2012A": ["bpic2012a", "bpi2012a"],
    "BPIC2012O": ["bpic2012o", "bpi2012o"],
    "BPIC2012W": ["bpic2012w", "bpi2012w"],
    "Hospital-billing": ["hospitalbilling"],
    "Receipt": ["receipt"],
    "Sepsis": ["sepsis"],
    "Helpdesk": ["helpdesk"],
    "Production": ["production"],
    "Road_Traffic_Fine": ["roadtrafficfine", "roadtrafficfines"],
    "Service-process": ["serviceprocess"],
}

ACTIVITY_COL_CANDIDATES = [
    "concept:name",
    "Activity",
    "activity",
    "ACTIVITY",
    "task",
    "Task",
    "event",
    "Event",
    "name",
    "Name",
]


def normalize_space(text: object) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.strip())


def normalize_key(text: object) -> str:
    return normalize_space(text)


def normalize_token(text: object) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_normalized_mappings() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for dataset, mapping in ACTIVITY_MAPPINGS.items():
        norm_map: Dict[str, str] = {}
        for raw, mapped in mapping.items():
            key = normalize_key(raw)
            if key in norm_map and norm_map[key] != mapped:
                raise ValueError(
                    f"Dataset {dataset} has duplicated normalized key {raw!r} -> {mapped!r} "
                    f"conflicting with {norm_map[key]!r}"
                )
            norm_map[key] = normalize_space(mapped).lower()
        if len(norm_map) != len(mapping):
            raise ValueError(f"Dataset {dataset} has non-unique normalized raw activity names.")
        if len(set(norm_map.values())) != len(norm_map):
            reverse: Dict[str, List[str]] = {}
            for raw, mapped in norm_map.items():
                reverse.setdefault(mapped, []).append(raw)
            duplicates = {k: v for k, v in reverse.items() if len(v) > 1}
            raise ValueError(f"Dataset {dataset} has non-unique mapped activity names: {duplicates}")
        out[dataset] = norm_map
    return out


NORMALIZED_ACTIVITY_MAPPINGS = build_normalized_mappings()


def resolve_dataset_name(name: str) -> str:
    key = normalize_space(name).upper()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if name in ACTIVITY_MAPPINGS:
        return name
    raise KeyError(f"Unknown dataset name: {name}")


def detect_dataset_from_path(path: Path) -> Optional[str]:
    full = normalize_token(str(path))
    matches: List[tuple[int, str]] = []
    for dataset, hints in DATASET_PATH_HINTS.items():
        for hint in hints:
            if normalize_token(hint) and normalize_token(hint) in full:
                matches.append((len(normalize_token(hint)), dataset))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    for alias, dataset in DATASET_ALIASES.items():
        token = normalize_token(alias)
        if token and token in full:
            return dataset
    return None


def read_event_data(input_path: Path) -> pd.DataFrame:
    name = input_path.name.lower()
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(input_path, sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(input_path, low_memory=False, encoding="latin-1")

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)

    if suffix == ".xes" or name.endswith(".xes.gz"):
        if pm4py is None:
            raise ImportError(
                f"Reading XES requires pm4py. Please run: pip install pm4py\nFile: {input_path}"
            )
        log = pm4py.read_xes(str(input_path))
        return pm4py.convert_to_dataframe(log)

    raise ValueError(f"Unsupported input format: {input_path}")


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def choose_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(input_root)
    if input_path.name.lower().endswith(".xes.gz"):
        stem = input_path.name[:-7]
    else:
        stem = input_path.stem
    return (output_root / rel.parent / f"{stem}.csv").resolve()


def detect_activity_column(df: pd.DataFrame, dataset: str, user_col: Optional[str] = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise KeyError(f"Activity column not found: {user_col}")
        return user_col

    mapping_keys = set(NORMALIZED_ACTIVITY_MAPPINGS[dataset].keys())
    ranked: List[tuple[int, int, int, str]] = []

    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        values = {
            normalize_key(v)
            for v in series.dropna().astype(str).tolist()
            if normalize_key(v) != ""
        }
        if not values:
            continue

        overlap = len(values & mapping_keys)
        if overlap == 0:
            continue

        exact = 1 if values.issubset(mapping_keys) else 0
        preferred = 1 if col in ACTIVITY_COL_CANDIDATES else 0
        ranked.append((exact, preferred, overlap, col))

    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][3]

    for col in ACTIVITY_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise KeyError(
        "Could not detect the activity column automatically. "
        "Please pass --activity-col explicitly."
    )


def replace_activity_names(df: pd.DataFrame, dataset: str, activity_col: str) -> pd.DataFrame:
    dataset = resolve_dataset_name(dataset)
    mapping = NORMALIZED_ACTIVITY_MAPPINGS[dataset]
    out = df.copy()

    raw_values = [normalize_key(v) for v in out[activity_col].tolist()]
    unknown = sorted({v for v in raw_values if v and v not in mapping})
    if unknown:
        raise ValueError(
            f"Dataset {dataset} has activities not covered by the direct mapping: {unknown}"
        )

    out[activity_col] = [mapping.get(v, normalize_space(v).lower()) for v in raw_values]
    raw_unique = {v for v in raw_values if v}
    mapped_unique = set(out[activity_col].dropna().astype(str).tolist())
    if len(raw_unique) != len(mapped_unique):
        raise ValueError(
            f"Dataset {dataset} violates one-to-one mapping. "
            f"raw={len(raw_unique)}, mapped={len(mapped_unique)}"
        )

    return out


def iter_supported_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in {".csv", ".parquet", ".pq", ".xes"} or name.endswith(".xes.gz"):
            yield path


def process_one_file(
    input_path: Path,
    output_path: Path,
    dataset: str,
    activity_col: Optional[str] = None,
) -> dict:
    dataset = resolve_dataset_name(dataset)
    df = read_event_data(input_path)
    col = detect_activity_column(df, dataset, activity_col)
    out = replace_activity_names(df, dataset, col)
    write_csv(out, output_path)

    raw_unique = sorted({normalize_key(v) for v in df[col].dropna().astype(str).tolist() if normalize_key(v)})
    mapped_unique = sorted(set(out[col].dropna().astype(str).tolist()))

    return {
        "dataset": dataset,
        "input": str(input_path),
        "output": str(output_path),
        "activity_col": col,
        "raw_activity_types": len(raw_unique),
        "mapped_activity_types": len(mapped_unique),
    }


def process_all(raw_root: Path, output_root: Path, activity_col: Optional[str] = None) -> List[dict]:
    reports: List[dict] = []
    skipped: List[str] = []

    for input_path in iter_supported_files(raw_root):
        dataset = detect_dataset_from_path(input_path)
        if dataset is None:
            skipped.append(str(input_path))
            continue

        output_path = choose_output_path(input_path, raw_root, output_root)
        report = process_one_file(input_path, output_path, dataset, activity_col)
        reports.append(report)
        print(
            f"[OK] {dataset:<18} | {input_path.name} | "
            f"activity_col={report['activity_col']} | "
            f"types={report['raw_activity_types']} -> {report['mapped_activity_types']}"
        )

    if skipped:
        print("\n[WARN] The following files were skipped because the dataset name could not be inferred from the path:")
        for item in skipped:
            print(f"  - {item}")

    if not reports:
        raise RuntimeError(
            f"No supported dataset files were processed under {raw_root}. "
            "Please check file names/folder names, or use --input and --dataset for single-file mode."
        )

    summary_path = output_root / "activity_mapping_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print(f"\nSummary written to: {summary_path}")
    return reports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct one-to-one activity replacement for the 20 PPM datasets."
    )
    parser.add_argument("--raw-root", type=str, default="./raw_dataset", help="Root folder containing all raw datasets.")
    parser.add_argument("--output-root", type=str, default="./mapped_dataset", help="Output folder for mapped CSV files.")
    parser.add_argument("--input", type=str, default=None, help="Single input file. If set, only this file is processed.")
    parser.add_argument("--output", type=str, default=None, help="Single output CSV path for --input mode.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for --input mode, e.g. BPIC2020I.")
    parser.add_argument("--activity-col", type=str, default=None, help="Activity column name. Default: auto detect.")
    parser.add_argument("--print-split", action="store_true", help="Print the recommended pretrain/test split.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.print_split:
        print(json.dumps(RECOMMENDED_SPLIT, ensure_ascii=False, indent=2))

    if args.input:
        if not args.dataset:
            raise ValueError("Single-file mode requires --dataset.")
        input_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve() if args.output else input_path.with_suffix(".mapped.csv")
        report = process_one_file(input_path, output_path, args.dataset, args.activity_col)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    process_all(raw_root, output_root, args.activity_col)


if __name__ == "__main__":
    main()
