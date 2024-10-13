limit_config = {
    # 样本内路径：/data/group/800080/AlphaDataBaseForZT_Cut
    "daily_path": "/data/group/800080/Apollo/AlphaDataBase",
    # 样本内路径：/data/group/800080/PanelMinDataForZT_Cut
    "minute_path": "/data/group/800080/PanelMinDataForZT",
    # 样本内路径：/data/group/800080/warehouse/insample
    "h5_path": "/data/group/800080/warehouse/prod",
    "factor_path": "",
    "factor_classes_dir": "/tmp/",
    "log": "logger.txt"
}

mapping_h5 = {
    'WIND_AShareBalanceSheet': 'ANN_DT',
    'WIND_AShareCashFlow': 'ANN_DT',
    'WIND_AShareIncome': 'ANN_DT',
    'WIND_AShareProfitExpress': 'ANN_DT',
    'WIND_AShareProfitNotice': 'S_PROFITNOTICE_DATE',
    'WIND_AShareFinancialIndicator': 'ANN_DT',
    'WIND_AShareTTMHis': 'ANN_DT',
    'WIND_AShareANNFinancialIndicator': 'ANN_DT',
    'WIND_AShareIssuingDatePredict': 'ANN_DT',
    'WIND_AShareDividend': 'ANN_DT',
    'WIND_AIndexFinancialderivative': 'OPDATE',
    'FDD_CHINA_STOCK_QUARTERLY_WIND': 'stm_issuingdate',
    'WIND_FinNotesDetail': 'OPDATE',
    'WIND_AShareIBrokerIndicator': 'ANN_DT',
    'WIND_AShareInsuranceIndicator': 'ANN_DT',
    'WIND_AShareBankIndicator': 'ANN_DT',
    'WIND_Top5ByLongTermBorrowing': 'ANN_DT',
    'WIND_AshareOtherreceivables': 'OPDATE',
    'WIND_AShareFinancialExpense': 'ANN_DT',
    'WIND_AshareInventorydetails': 'OPDATE',
    'WIND_AshareFinancialaccounts': 'ANN_DT',
    'WIND_Top5ByAccountsReceivable': 'OPDATE',
    'WIND_AShareAuditOpinion': 'ANN_DT',
    'WIND_AShareSalesSegment': 'OPDATE',
    'WIND_Top5ByOperatingIncome': 'ANN_DT'
}

overwrite_h5 = ['WIND_AShareIndustriesClassCITICS', 'WIND_AShareDescription', 'WIND_AShareIndustriesCode',
                   'WIND_AShareST',
                   'WIND_AShareCapitalization', 'WIND_AShareFreeFloat', 'WIND_AShareIPO', 'WIND_AShareAgency',
                   'WIND_AShareCOCapitaloperation', 'WIND_ASharePledgepro', 'WIND_AshareStockRepo',
                   'WIND_AShareCorporateFinance',
                   'WIND_AShareIssueCommAudit', 'WIND_AShareEquityDivision', 'WIND_AShareStaff',
                   'WIND_IPOCompRFA', 'WIND_IECMemberList', 'WIND_AShareLeadUnderwriter',
                   'WIND_AShareRightIssue', 'WIND_AShareSEO', 'WIND_IPOInquiryDetails',
                   'WIND_AShareManagement', 'WIND_AShareIncDescription', 'WIND_AShareIncQuantityPrice',
                   'WIND_AShareIncQuantityDetails',
                   'WIND_AShareIncExercisePct', 'WIND_AShareIncExecQtyPri', 'WIND_AShareEsopDescription',
                   'WIND_AShareEsopTradingInfo',
                   'WIND_AShareStaffStructure', 'WIND_AShareMajorHolderPlanHold', 'WIND_AShareTypeCode',
                   'WIND_htzqedbdzzbs',
                   'WIND_AShareMainandnoteitems', 'WIND_AIndexMembers', 'WIND_AShareConseption', 'ETC_CHINA_STOCK_WIND',
                   'CALENDAR_CHINA_STOCK_DAILY_HTSC']
