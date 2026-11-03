"""Ownership, dependency, transport and asset guards for Stage8A1.

Contract fingerprints come from 989f27594ec08a4cd1b6f1266e565ba34cee62a1.
A9 updates only the five accepted six-field result shapes and retires the confidence
enum. All other fingerprints retain that baseline. Chemistry expectations below
use explicit methane atoms and independent charge/electron/spin accounting.
"""

import ast
import hashlib
from importlib import import_module, resources
from importlib.util import resolve_name
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


_RULES = "hotpot.cheminfo.rules.chemistry.scientific_state"
_OPERATIONS = "hotpot.cheminfo.operations.calculation.scientific_state"
# This file is also collected against an installed wheel outside the checkout.
_PACKAGE_ROOT = Path(import_module("hotpot").__file__).resolve().parent

_CLASS_CONTRACTS = {'rules.charge.contracts:NetChargeCandidateEvidence': 'c8e60feecbe1356a7420b4fb27d64aab55e40d2d1722a9162159e7da01ba5028',
 'rules.charge.contracts:AtomFormalChargeEvidence': '16fcdb42a448f4a8506b216e01b969c818484e5bb21655145f8a141abd4da992',
 'rules.charge.contracts:ComponentChargeEvidence': '856c24af71bcfa3a3592ae29ff27624df5a7c4437e4be5fc28515356a0e4edc6',
 'rules.charge.contracts:NetChargeResult': 'a1b560f08cb74a761b23d85a2a39e64a12f80228f5104d467925bc491740b482',
 'rules.charge.contracts:LewisFormalChargeCandidate': 'e7563559e035c07b6c54edd2aeb36db3825fddfae5e6041944fc3b8cf7b492ec',
 'rules.charge.contracts:LewisFormalChargeResult': 'aa4ede290c60ed09c5f306b060456d9ff84129c627a91b7c284fbee60628c126',
 'rules.electronic_state.contracts:SpinContributionScope': '90c26cd1396886e1cf4c196dea7b57e3a68f59c7f2990326177a3fc9ae0aac17',
 'rules.electronic_state.contracts:ElectronicStateCandidateEvidence': 'b66bbf1d538288e8825d69f6622833b3f82ac2341708a3e6e9312fc2135ca24f',
 'rules.electronic_state.contracts:SpinContributionCandidate': 'cf8d56d233fb85bc6b02b350e3a552b29d14b84351424d3d5f8663f7ee4dac82',
 'rules.electronic_state.contracts:ElectronicStateCandidate': 'c1ec630768375e8800c5cfa7d553d825b363038e6fa1a796949affd7dc608a26',
 'rules.electronic_state.contracts:ElectronicStateInferenceResult': '1f9b53890e8eed925721ddbbd4c49855b9a160545b689222eb89ff5baab47e04',
 'rules.electronic_state.contracts:NeutralAtomReference': '5762aa43b5d713ca0f334e3921c56c2f14f89e8252c4d3d7f6f949b28c42b16c',
 'rules.oxidation_state.contracts:CountingConvention': 'dfa74e56f6d3286ec909a9fdc795e2ba7eff2f24d4b79141a64dca89c412d6a6',
 'rules.oxidation_state.contracts:OxidationStateOverride': '367439a5eadfb6709ce60398a2b4348a12b0fa27c70b1b1d106ebcf72cf26511',
 'rules.oxidation_state.contracts:CoordinationFragmentAssignment': 'f3fe723dea9407728d6d7a0f3389050f51c86a9ad752641eb753f5718e970d86',
 'rules.oxidation_state.contracts:CoordinationCountingCandidate': '04b408d6aa3ec04d32c38290cbf7afbf79c356bf3aa51843f157c35010f2a91f',
 'rules.oxidation_state.contracts:CoordinationElectronCountingResult': '8655a9417e042560b51791d258307200ddfe8be4bd8bb045c04ae68ca16e3871',
 'rules.oxidation_state.contracts:OxidationStateCandidateEvidence': 'f30e282e53f2383d79107b0fd5e9ec48fbb37b25172dd8dd4eabe96713f9a221',
 'rules.oxidation_state.contracts:OxidationStateInferenceResult': '4a82495782ec26846ca3e44f4a35465f30429bcbdc20a5c5ea74ca0ef199101a',
 'rules.oxidation_state.contracts:ElementOxidationStateDomain': 'e3048a8f9d40780ffc493738f302852fc7c203899ec1fe957e6af168267a89db',
 'rules.legality:_HardRuleAssessment': 'e533716f4789ff04f1cf0f0a6d7a04792aa27e32db22823515a3c0e3ddf6468a',
 'rules.composition:CandidateDomainScope': '64858839159e2a31f1d2990f70874386cde1a17fe873b93c0773fa2f71799226',
 'rules.composition:MaterializationStatus': 'dd9b146b22e04a7ba106994cb283c1759b93471e1fe2d604438be2831537be6e',
 'rules.composition:CandidateSearchStatus': '8fbbda34332a81e0cc5bf0793c41dbf82d55da00cf311780a309715e4de55fdc',
 'rules.composition:StateValueOrigin': 'f72df1fea44b82cb0fda8d2fa3572d5a188e78523e7632b7f9b5cfaac44a23b7',
 'rules.composition:ScientificStateIdentity': '996f40d4d68b72b28733cb67a3b2e8f372456914c9474e1712cfe19e8577e205',
 'rules.composition:FactorKey': '42682a85ff58fc0a71f1b9e09077fb2c1a10a4f6d1a51dd97a3b6a045e4fec44',
 'rules.composition:CandidateConstraint': 'feefc3a6cf54520c50dbdf8e1dba2c783efbc17be7081d1fa958dfd6fe49c597',
 'rules.composition:FrontierBound': '8c21d6a2973fc4da37c054b9cfc4e01b1a2895b7e311532b9b3f11cbaf1cc676',
 'rules.composition:CandidateDomainSummary': 'b6462739c98fbd85035fc4b239cf1abf4437b472c1fd820a571ec3a0f1fda535',
 'rules.composition:RejectedPathSummary': 'ee5d4c8bfffbef3b13f116a39898d35e02061fdfcb09f4c344b3c3b0ca3840a1',
 'rules.composition:StateValueRef': '54abeef8ba00a62df4cb32e219821b6a4b04f1275fc030577ed41c46eec78858',
 'rules.composition:DerivedStateValue': '8f4ba73c43ec52a44ed8f25980c9aad5e97f4ad6fdc2356ec1cb97b062dfb130',
 'rules.composition:BranchAssignment': '2c4e9ea49757abd945eaa3eaf63b80fe756f4a9994af1544bf89da62e8fb2c29',
 'rules.composition:FactorCandidateValue': 'd66759d320c4ea2152fb445480e67ebe5eb0d70c1aa1bef9e71860afb605b1c9',
 'rules.composition:FactorDomainContribution': '1f9c1924adf76de9d0ab4ef81b4c894cf0379ec9e4dc4c70c60a630384bc2ab6',
 'rules.composition:CandidateFactor': 'bf372d78c616b43db63eee20a6c38615f091d875b310243cf8ba0052836a9502',
 'rules.composition:ProposedCandidateValue': '194f09298745bc3181d55f2a1e76d078ef6d1d462153ff744339187162d76198',
 'rules.composition:ProposedCandidateFactor': '0b0f49b6e588a3ae3d9be9693a6df31631ed3e3ac1eb26fe9513d2815f0e5141',
 'rules.composition:ConstraintGuard': 'c8f3ea78618d102c0106fcd0ad907fc5a3890d86a3fb11af5b596a7bf562b705',
 'rules.composition:GuardedCandidateConstraint': '05402b22981ce117e1989feabe7f6d69f7a71c5c358906c758a0821c7b307d2f',
 'rules.composition:CandidateProposal': 'c86b751dc89505c0237dbbcde6f666bd59cd665a2e1992d692e391be663c464b',
 'rules.composition:CoupledScientificStateCandidate': '14f98badb598e4df259ab5d71b4dbca1f0caa1e59f40784473b8f39f71a32f54',
 'rules.composition:ScientificStateCandidateSpace': '9d8f6fa2982bb3a20d3d0ef1edc2f430115054fbf68c4faee0d5a8efbd77976a',
 'rules.composition:ScientificResolutionResult': 'cd92ed27711cca61a6aaa5b2b71e1f70fb5f6e6b31fde796e88780dec01f5fa7',
 'operations.composition.decision_search:_DecisionRelation': '72d75dbb6ead4151fb517ba4df8385a08f36c02da2b2ebe983a230d69ecce423',
 'operations.composition.decision_search:_AuthorityVector': '3d552b27b6afbeb880f4eb69375b9a8640d9e9308b848f5097fdb80d5c6fe631',
 'operations.composition.decision_search:_PreferenceKey': '4a82af7c1c3f812eafbc7a364d5c6d2ba24709d21dc2377dc61c90a57e60e994',
 'operations.composition.decision_search:_DecisionEquivalenceKey': '160dc8f18c32d01f68da927e7d13ba416a1892ef8803b772c25312677b6fcfec',
 'operations.composition.decision_search:_DecisionRecord': '0353ad03fead1264ca9b501289985c26f54ce940c77478940c6eac65b9696e47',
 'operations.composition.decision_search:_DecisionBound': '9a5ec436c60080b8e1593f25678aedf9053c3ce0092a3c6e97c2a7b56df4e21d',
 'operations.composition.decision_search:_DecisionSearchCompiler': '1910d9c28fc0e876c1fc6d606a8845ba505f873443631ee5f70426b7f73a790d',
 'operations.composition.expand:_SelectedValues': '653535c26ae84af0e581091ba40bdec593c26776cfe28edfe8d11fb239881d46',
 'operations.composition.expand:_SearchNode': 'd89d19ef7fe36cf7b35896edcedb9929599a824ad5687953e0fe2cefae7441c0',
 'operations.composition.expand:_FrontierEntry': 'e055df649be473479562396db2694a4e80172aa28e98a58a7d0f586567a4cc8a',
 'rules.providers.americium_curium_iii:AmericiumCuriumIIIPriorRule': 'e1de491262de10bc3c8690c6a7e6386ca559872c8b843cb6c4af2231374e5bbf',
 'rules.providers.neutral_l_donors:_AtomSignature': 'fe3f6071f56e2b5c99e0472b6d1c3a32188e0876d8fb110eee35fba33381b7d7',
 'rules.providers.neutral_l_donors:_ComponentSignature': '9d5090b091df9b75b0f3b6225d5fc4d94f8cef3e5029b07386c10ade989bf277',
 'rules.providers.neutral_l_donors:_DonorRegistry': '8d6c9aeb56a90c4bcf7a61cea74d1af54586dad73887e37217c3ce21ba8b9d51',
 'rules.providers.neutral_l_donors:NeutralLDonorRule': '33e02ef8c75f6322a5531f52672728217cfbec6bf87ca0f3925165566ac5556c',
 'rules.providers.ionic_configuration_spin:_IonicConfiguration': '3716ab296c49c0010f2f3d3ba763cb39cb6deb16e532393b93f905cf25058c08',
 'rules.providers.ionic_configuration_spin:_IonicConfigurationRegistry': '98005146259721331069f6b6862c1f09e635cd018a1e620aff1bce4f95e64a71',
 'rules.providers.ionic_configuration_spin:IonicConfigurationSpinRule': '7f7356a2582a96505a776557b786349446b8ad6d50c50d881dc5603245247901',
 'rules.providers.contracts:RuleKind': '4589fdb4492e3214f699f889765709bea50eb8e6cf766337751689f41182e381',
 'rules.providers.contracts:RulePhase': 'e74d7b88a219996e6ffb66e7c43242e0e0bd9e9e6665d107c68068feb1e9b96d',
 'rules.providers.contracts:ResultType': 'ed16090e8f2a0e8245a3b101663e0759cfd3b9bc9e203e1c79fc53e60ffde110',
 'rules.providers.contracts:RuleDependency': '3474197e9463fa26a1d99f06b7f7621b55e59d7390cb70b202a2cbe6ca8bb6ae',
 'rules.providers.contracts:RuleSpecification': '60ba7294480d0e4d54c8096e69322ad500c3f06d640551e3210c35c1e2eef091',
 'rules.providers.contracts:RuleToggle': '6a30973fe2cadd2863fad77d3ec3bdb73c9ae299a6405b6156f483e657ad7af7',
 'rules.providers.contracts:EvidenceItem': '0bd918b041cf67ee4f07bcfecdff2e2cd21a451d7badff2e82e953a3e1863322',
 'rules.providers.contracts:RankAdjustment': 'a100162ccc9948fcb8eb39079bb47430573d7da067c0367bfade39a9ccd531a1',
 'rules.providers.contracts:RuleApplicability': 'db6ebdc52a60fb84226ac1116ee63ddb3de46bd5456ebc5e1b38e960ebd8864b',
 'rules.providers.contracts:RuleContext': '3a5413f84371987a0574b968e38120deee4ef8aa1ab5c403f6a8440b026e71c9',
 'rules.providers.contracts:RuleOutcome': '85e2f9c9168ee4c1fdb5d56d95b286e582c067b044e78d450a5b5284284cefb6',
 'rules.providers.contracts:RuleTrace': '07a2f121276999b920b83f02a5e2474b371f7fc3d5c630e1597ba100abf08731',
 'operations.contracts:ActivatedRule': 'eb49f3fb229879740965a58de220f902882bdb90d726f6a3d2d35d3ff1235a26',
 'operations.contracts:RuleSet': '52cfc1918ff379bff45c02f9bba6fef48b77a71209fadea8d10df76447f5230d',
 'operations.profiles:_ScientificStateProfileArtifact': 'b53736aceabd183e0c1b63b3117671d4198f81989cc768ec8f0598542d558fc3',
 'rules.providers.lanthanide_iii:LanthanideIIIPriorRule': 'e1de491262de10bc3c8690c6a7e6386ca559872c8b843cb6c4af2231374e5bbf',
 'operations.rule_controller:_RuleExecution': '81c9756048da5a7da5ccb664b2e847da6c34b3fa2fa79e64c2cdf633a6300cff',
 'rules.providers.monoatomic_ions:_IonEntry': '9e722f4041967647f50fda4d2c842c0b02c2b98fa740585eb0a92f03553cdb87',
 'rules.providers.monoatomic_ions:_IonRegistry': '1c07c255dc3d7c9f57dda5815e1db160d37f4bad606e05fb8a36e4ee9708bda2',
 'rules.providers.monoatomic_ions:MonoatomicIonRule': '18606cc08d14dd683e71a7ea384cacc7968424827a41a908ff3c1364aa914fb4',
 'rules.providers.f_block_hund:_HundPriorRegistry': '2e499587bb5ba7398f41cfd68590d8252e70febc4b0a361b7b1e0f75c9465e6c',
 'rules.providers.f_block_hund:FBlockHundPriorRule': 'd331dd92ece1d3880a373b26b2deec5a2fc15875a90366cdc6be42ec599e32bf',
 'rules.providers.main_group_closed_shell:_ShellDomain': 'b4a0ab50eb3f6a094ae8141622107ef96b69c3b9feb113e872ae70be67358d10',
 'rules.providers.main_group_closed_shell:_ClosedShellRegistry': '4b018d571ffb5c11d4c3d6a7ae3e34b54287768d58347f1604a9f99f7d17ffc9',
 'rules.providers.main_group_closed_shell:MainGroupClosedShellRule': '8b40053d9e965d5e6a8c1fa0496142f9c31455db1bb8ce095d1429d0b4b58fae',
 'rules.providers.base:InferenceRule': '381fc55a448b97ccf6ce23842239c60e297bcb1d470dbce899a3c48b63c4a930',
 'rules.providers.oxidation_priors:_ElementOxidationPrior': '9f3bb93eb0231c5f8c8b815484caa1437f75343c36a6bab7f796c48655e6d5ea',
 'rules.providers.oxidation_priors:_OxidationPriorArtifact': '0dec6ea05cba7bf0c27a9d9af16da6cc7d95c3de505fcdc3141a2b574bd70ef1',
 'rules.providers.oxidation_priors:_OxidationStatePriorRule': 'cd31380dbd499341d01e5ee9223c36b78c129b06145009f446e12377a1681895',
 'rules.providers.anionic_x_ligands:_XLigandRegistry': '31c11cfafb64282f207c276a86bf1057a15191e8689b2ee66712c059a788fbd7',
 'rules.providers.anionic_x_ligands:AnionicXLigandRule': '33e02ef8c75f6322a5531f52672728217cfbec6bf87ca0f3925165566ac5556c',
 'operations.rule_registry:InferenceRuleRegistry': 'e370a2adb34f4df514e8b03a9d7f80ef7cccc1d2c43eb4c0caec38f6e7d23ee0',
 'operations.manifest:_ScientificStateRuntimeManifest': '5e6d639f3cbbf6501e136b8682865e3f0a2e0cd8dd3c7e8346113228d836685e',
 'rules.providers.trusted_history:TrustedScientificStateRecord': '58697fd7f0b75a42a56950526c11b9440f80fca55624ebbda6260bf97eea69f8',
 'rules.providers.trusted_history:TrustedHistoryRule': 'd12f8571f695eab4d2a4716bd7bcb5da43e6d98775bb86410b1a1f4bb6fd84f9',
 'operations.profile_decision:_AssumptionSeverityPolicy': 'd9e737e7546bdc82d6faec43801e4bff3ce1093e1f7bf24fba1b13d564c93007',
 'operations.profile_decision:_CoordinationScreeningDecisionCompiler': 'e892d16a82787dd3065f0019b1848da5e0da9918e19f0e75d229145a5bf775ff',
 'rules.providers.thorium_iv:ThoriumIVPriorRule': 'e1de491262de10bc3c8690c6a7e6386ca559872c8b843cb6c4af2231374e5bbf',
 'rules.request:StateRequest': '3fae8b1a2120f3237c1e47e3f5422dd31b75bf2a9ec529577bd277bd92c88286',
 'operations.state_resolver:ProvidedScientificResults': '93541f964766a51f96a12456ec73fa83a84d3b822ce1a9486aba1eb926ea0a8c',
 'operations.state_resolver:ScientificStateBundle': 'b97150e3bb8cc87fbe90f86b834702ffe26575ba7ecba698fa9eb0947ecf072c',
 'operations.state_selection:StateDecisionAction': 'e7b2a79027007be7760035f5d47e73ca518e95aec1aebbba3f3a5825351ea595',
 'operations.state_selection:StateDecisionSource': '452f9c1b6bf9f07bda546b1141cd0b603d645c28319ee18acd60818b2a2db0ad',
 'operations.state_selection:DecisionCertificationStatus': '740a96aa2ffaf2b2169a38750badd26e43f5eb1761c5b2aa64d14cad6c62e94d',
 'operations.state_selection:StaticEligibilityFinding': 'ac6d82854c6751b6ed2fb96c66260189c7fe62fbaf13b86e2c9b4f4a841ec643',
 'operations.state_selection:DecisionReason': '58bad2f439366415f08e54d735bb0b911cc42f6c8d19947fafd0efa19a72b505',
 'operations.state_selection:StaticProfileEligibility': '7c1b67e4a72922a7f0206ea3ad70be6a491a4bb8a4908083768de9a694cb9079',
 'operations.state_selection:DecisionCertifiability': 'd75dbc6d1fe0fd15a2168ce22df70b9892806df211a1fed8f1fc20b73a0f80cc',
 'operations.state_selection:ExecutionStateDecision': '7165091243445f5fdea689506c25114a1b27bb3aaf8ebd4ab9c97bcaa5b33b72',
 'operations.execution_selection:BackendStateValues': 'f909c0a9537ffc28f1570996535656b9e8afb0e52ef0b7fd2a7275c34fba8108',
 'operations.execution_selection:BackendStateRequest': '9ce591a759e2ef1aa4a0d90555423d6fb33ee560eacf84bebfd1a0c207b37fb5',
 'operations.execution_selection:BackendDecisionSource': '42b16d90116c9f41e161fed09fe91b0f89c817fcfd0e8929f324a187556fcf95',
 'operations.execution_selection:BackendExecutability': 'df15c4e1afa6be26216a49d4d8604587692053a2507570b2dd671f915999b010',
 'operations.execution_selection:BackendProjectionFinding': '58bad2f439366415f08e54d735bb0b911cc42f6c8d19947fafd0efa19a72b505',
 'operations.execution_selection:BackendExecutionDecision': '92198c82bbdc3d21cb7f9dc3a1acade32df55c45a7514f3467afc8c2cf7bbd27',
 'operations.execution_selection:BackendExecutionEvidence': '6d2d646c784b9b2d72ad03bdf9f6838a0457cb5aae918027f263c7eff8aa3a24',
 'operations.execution_selection:ExecutionPolicySet': 'e2bf3fc258b516529f1e23d240c95665bbb894d0cbe50b45f7d7e42250fa3bf5',
 'operations.execution_selection:BackendStateProjection': 'f86e9a58ee115fb5bf86f46e3fc1ee2a839796316dcc6acb178bb2830b6cad36',
 'operations.execution_selection:BackendProjectionStatus': 'f3c29f3d0b42eb034e482833b2d525d7c854265fcfb8502905071ec1b04d1264',
 'operations.execution_selection:BackendProjectionResult': '3ed488d6f30e90a8af0b44a7e5f1c12f4f6971cea42c941d1649fcc6c744ee39',
 'operations.execution_selection:ScientificStateWorkflowResult': 'ec3c83fe891299e6bb8c55824b1f9097de12302dbd1835722b754873d0170e8d',
 'rules.contracts:ScientificQuantity': '50331c435b0c943d28e7929ee7157213d0d630355df0567ff425b1c4bd740cad',
 'rules.contracts:ScientificAdmissibility': '3c32560a85691b11992ac4ef89370094282904343b3c95c72ef0699fe23835ec',
 'rules.contracts:ResolutionStatus': '942dfffc913e17b794c66ed3842ed141c6d148e8625364a63f56a8f46197a1c7',
 'rules.contracts:CoverageStatus': '2c270492499469647cdb3b74fb26c51fb21aa279752be741527927aae7d2acbe',
 'rules.contracts:StateValueAuthority': '723bf691b93bbca3a97a3bff07c172ee3c3b20dc32ff76e652fff4ac54a83625',
 'operations.contracts:StatePolicy': 'ae425b8ae000b540b7e1adb5c3dc8b115f609358d08905770bf3f675da1fefff',
 'structure.snapshots:MolecularFactsIdentity': '5a940d328bbc5147db96e3385f0a69033b4e9a8cf23398f1ac34360077e7f955',
 'adapters.calculation.contracts:BackendIdentity': '996f40d4d68b72b28733cb67a3b2e8f372456914c9474e1712cfe19e8577e205',
 'structure.snapshots:ComponentIdentity': '59b5cb4ea045296f54da674207e56dcc2f394a055c0d8b0620951804127bdded',
 'structure.snapshots:HydrogenInventory': '3549203cfc6a613bcc37d256e6c337d0151b90f1d60cc344d9c7755bf3a1a44e',
 'structure.snapshots:SemanticBond': '9344fdb15fb75d8d9160134a33da975d4d9a55dc7e4874c16aadcfad29a8c03c',
 'rules.contracts:ScientificFinding': '58bad2f439366415f08e54d735bb0b911cc42f6c8d19947fafd0efa19a72b505',
 'rules.contracts:CandidateProvenance': 'f963ee73e566994a72ddc3120ae3b79243858a3e76210a0e297eb3128daf2284',
 'structure.snapshots:MolecularFacts': '9794e310ff58da4b8b8dcfa73656e551efc4ce0711f784d65e8c76f34fc76d5f',
 'operations.contracts:ProviderToggle': 'c7d74aeb3629f4a2cb185c2280b68ec8a646a4182523a87af8ec8c7d288572cc',
 'operations.contracts:ProviderSet': '7304d4192dacb72d49a77781d5340e01508f51f47a69d0a68cdcf34c9df292e5',
 'rules.contracts:CandidateBudget': 'd2344ad7f99bc4562fdfc8845808de97c96dc455474935335fb3d392c6c5a798',
 'rules.contracts:ScientificOperationRequirements': 'a07b26fb4aa5a5139f38a21d79877963de8520224ed84b48b72d645ecb6be8ee',
 'adapters.calculation.contracts:BackendContext': 'e3735dbe3dead04d033b43bf0abb1b3edbdae3b2a1ca41be213428db2ac0cdfc'}

_FUNCTION_CONTRACTS = {'rules.charge.lewis:solve_main_group_lewis': '3ca2eae937241cabaa1ebb61cefb45096200a0648c738b090836172ed9c19828',
 'rules.charge.net_charge:resolve_net_charge': '1f9a7d9cb770881a56a96fd2632937862cc515675a1bb65138ad69b5445917c4',
 'rules.electronic_state.neutral_atom:neutral_atom_reference': '3398fca98cad7a26778f507d79d400ce9b5bc3c9acd4e60cc430db9e90f90ea8',
 'rules.electronic_state.inference:infer_electronic_states': 'd526d30a68ad35fdc0ed78a980b1517d1f3f5862fd20136c323f012c8d0e2a63',
 'rules.oxidation_state.coordination_counting:infer_coordination_electron_counting': 'b27d7697cae77c24a5c567b44b4f24f31a2b7ea146f8120cd66fd9e46f3aab18',
 'rules.oxidation_state.registry:element_oxidation_state_domain': '61ed3a6583df32f043ae828d18d01f6be084c8835fde98a1e845b154fd11d373',
 'rules.oxidation_state.inference:infer_oxidation_states': '122b06513028db0ad748f9287d7b36af774f644a2568dfad4ddac7dc8e308ffd',
 'operations.composition.compose:compose_scientific_state_space': 'f324f7572d029f55131e95a6f5578bf10dfe88a4f30dd951bafd81b08e837fc0',
 'operations.facts:extract_molecular_facts': '9e252bb2ba22143644108ea3da4239faf6a75f2f0731f51bbb651da3e0373846',
 'operations.state_resolver:resolve_scientific_state': '1848cacf52631ed29b91d5bc36fd1a24630b538de4e0500e9e2532295d2ce73b',
 'operations.execution_selection:project_state': 'bd63f6795b2c591e5f5b6796cb6596fc7d01bfc07ba5451bec44b8f34f5dce44',
 'operations.execution_selection:resolve_and_project_state': 'e1a1f89fb418368b8297eeeba1e5228c6372cf400d8424f486a739867ce98d7a'}

_RESOURCE_DIGESTS = {'neutral_atom_references.json': 'ef504d3d5e2429623484a0b8935c46698d72a78a2f6a2b82e08afa5d44c4a3cf',
 'element_oxidation_states.json': 'bfb1b9f8fb58ba96a6b328d7a6b17f050ec45cbba2596d1219d8b49c20763f3a',
 'coordination_screening_v1.json': '574ed32272b9af9ac3f1f2ff879082bd37ba78e4d67f3305a46bf8091bff8ab5',
 'lanthanide_iii_prior_v1.json': '4acdc05ddc3ed9742b84abf25674c9db659302d14640f1608df49a80ab7e33bb',
 'americium_curium_iii_prior_v1.json': 'e9a6574effd8b371c08a8d9397d85e47e4919a6056fddbe575dcddde7addb3eb',
 'main_group_closed_shell_v1.json': 'e16c36833aec45c88c0e40f202ed62e2ad6c9972d9570c7ebd9992a2c2645280',
 'anionic_x_ligands_v1.json': '38aac2fed04dedfb2203a0b7efa8551d3b7f85af2ae01af479554f89a9eceb08',
 'ionic_configurations_v1.json': '908ccd570ec41a8f84794800a733a628a5382a5c55500d187722b57e903c9062',
 'neutral_l_donors_v1.json': 'bf75ff4947567513a40702140df042acbe4a7591fe7baf90bffa05caaa41429d',
 'f_block_hund_prior_v1.json': '5f38ab42daecfbf69f29917656a15ca3e6e98d52d7ea18e1bec5717998492d36',
 'thorium_iv_prior_v1.json': 'a274690c8db7782a8cd10c0919a0a1c941ab515d9f417c84949f640df022ef38',
 'monoatomic_ions_v1.json': '955dc5cbbd57a63545c12a7de537ad196548cec6a7851b0e3212212b9cc1ac0a'}


def _owner(key):
    owner, name = key.split(":")
    if owner.startswith("rules."):
        owner = _RULES + owner[len("rules"):]
    elif owner.startswith("operations."):
        owner = _OPERATIONS + owner[len("operations"):]
    else:
        owner = "hotpot.cheminfo." + owner
    return owner, name


def _source(module_name):
    path = _PACKAGE_ROOT.parent.joinpath(*module_name.split("."))
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__init__.py"


def _function_contract(node):
    return (
        node.name,
        ast.dump(node.args),
        ast.dump(node.returns) if node.returns else None,
        tuple(ast.dump(item) for item in node.decorator_list),
    )


def _class_contract(node):
    members = []
    for child in node.body:
        if isinstance(child, (ast.AnnAssign, ast.Assign)):
            # snapshots.py predates postponed annotations; quoting the relocated
            # annotations there preserves their former runtime string values.
            if (
                node.name in {"MolecularFactsIdentity", "ComponentIdentity", "HydrogenInventory", "SemanticBond", "MolecularFacts"}
                and isinstance(child, ast.AnnAssign)
                and isinstance(child.annotation, ast.Constant)
                and isinstance(child.annotation.value, str)
            ):
                child = ast.AnnAssign(
                    target=child.target,
                    annotation=ast.parse(child.annotation.value, mode="eval").body,
                    value=child.value,
                    simple=child.simple,
                )
            members.append(ast.dump(child))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.append(_function_contract(child))
    return (
        tuple(ast.dump(item) for item in node.bases),
        tuple(ast.dump(item) for item in node.keywords),
        tuple(ast.dump(item) for item in node.decorator_list),
        members,
    )


def _contract_digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def relocated_definitions():
    owners = {_owner(key)[0] for key in _CLASS_CONTRACTS}
    paths = {_source(owner) for owner in owners}
    for package in (_RULES, _OPERATIONS):
        paths.update(_source(package).parent.rglob("*.py"))
    definitions = {}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                definitions.setdefault(node.name, []).append((path, node))
    return definitions


@pytest.mark.parametrize(("key", "expected"), _CLASS_CONTRACTS.items())
def test_relocated_class_has_one_owner_and_unchanged_contract(
    key, expected, relocated_definitions,
):
    owner, name = _owner(key)
    definitions = relocated_definitions[name]
    assert len(definitions) == 1, (key, [str(path) for path, _ in definitions])
    path, node = definitions[0]
    assert path == _source(owner)
    assert _contract_digest(_class_contract(node)) == expected, key
    assert getattr(import_module(owner), name).__module__ == owner


@pytest.mark.parametrize(("key", "expected"), _FUNCTION_CONTRACTS.items())
def test_public_function_is_defined_at_its_owner_with_unchanged_signature(key, expected):
    owner, name = _owner(key)
    nodes = [
        node for node in ast.parse(_source(owner).read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(nodes) == 1
    assert _contract_digest(_function_contract(nodes[0])) == expected, key
    assert getattr(import_module(owner), name).__module__ == owner


def _import_targets(path):
    parts = list(path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts)
    package = ".".join(parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        # ast.walk deliberately includes imports beneath TYPE_CHECKING and functions.
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            yield target
            yield from (target + "." + alias.name for alias in node.names)


def test_pure_rule_import_closure_never_reaches_an_upper_layer():
    forbidden = (
        "hotpot.cheminfo.operations",
        "hotpot.cheminfo.adapters",
        "hotpot.cheminfo.core",
        "hotpot.cheminfo.structure.molecule",
        "hotpot.cheminfo.structure.atom",
        "hotpot.cheminfo.structure.bond",
        "hotpot.cheminfo.calculator",
    )
    pending = list(_source(_RULES).parent.rglob("*.py"))
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for target in _import_targets(path):
            assert not any(target == prefix or target.startswith(prefix + ".") for prefix in forbidden), (path, target)
            if target.startswith("hotpot"):
                dependency = _source(target)
                if dependency.is_file():
                    pending.append(dependency)


def test_scientific_leaf_initializers_export_contracts_without_eager_functions():
    for leaf in ("", ".charge", ".electronic_state", ".oxidation_state", ".providers"):
        module = _RULES + leaf
        imports = [
            node for node in ast.walk(ast.parse(_source(module).read_text(encoding="utf-8")))
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        assert all(
            isinstance(node, ast.ImportFrom)
            and resolve_name("." * node.level + (node.module or ""), module) == module + ".contracts"
            for node in imports
        ), module


def test_scientific_adapters_do_not_import_operations():
    for owner in ("hotpot.cheminfo.adapters.structure.scientific_state", "hotpot.cheminfo.adapters.calculation.contracts"):
        assert not any(
            target == "hotpot.cheminfo.operations" or target.startswith("hotpot.cheminfo.operations.")
            for target in _import_targets(_source(owner))
        ), owner


def test_scientific_calculator_implementations_are_removed():
    assert not list((_PACKAGE_ROOT / "cheminfo/calculator").rglob("*.py"))


def test_default_candidate_budget_remains_the_approved_bounded_work():
    from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import DEFAULT_CANDIDATE_BUDGET

    assert (
        DEFAULT_CANDIDATE_BUDGET.max_materialized_values_per_factor,
        DEFAULT_CANDIDATE_BUDGET.max_expanded_paths,
        DEFAULT_CANDIDATE_BUDGET.max_frontier_paths,
    ) == (64, 4096, 1024)


def test_all_twelve_resources_preserve_pre_migration_bytes():
    data = resources.files(_RULES).joinpath("data")
    pending = [data]
    actual = {}
    while pending:
        for child in pending.pop().iterdir():
            if child.is_dir():
                pending.append(child)
            elif child.name.endswith(".json"):
                assert child.name not in actual
                actual[child.name] = hashlib.sha256(child.read_bytes()).hexdigest()
    assert actual == _RESOURCE_DIGESTS


def test_neutral_reference_keeps_canonical_json_digest():
    from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.neutral_atom import neutral_atom_reference

    payload = json.loads(resources.files(_RULES).joinpath("data", "neutral_atom_references.json").read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    reference = neutral_atom_reference(6)
    assert reference.electron_count == 6
    assert reference.content_digest == hashlib.sha256(canonical).hexdigest()


@pytest.fixture(scope="module")
def methane_facts():
    from hotpot.cheminfo.structure.snapshots import (
        ComponentIdentity, HydrogenInventory, MolecularFacts,
        MolecularFactsIdentity, SemanticBond,
    )

    component = ComponentIdentity("explicit-methane:component", (0, 1, 2, 3, 4))
    return MolecularFacts(
        identity=MolecularFactsIdentity("explicit-methane"),
        geometry_digest=None,
        atomic_numbers=(6, 1, 1, 1, 1),
        mass_numbers=(None,) * 5,
        atom_map=(0, 1, 2, 3, 4),
        source_components=(component,),
        covalent_components=(component,),
        hydrogen_inventory=HydrogenInventory((4, 0, 0, 0, 0), (0,) * 5),
        semantic_bonds=tuple(SemanticBond((0, hydrogen), 1.0, False, "covalent") for hydrogen in range(1, 5)),
        coordinates=None,
        nuclear_charge_sum=10,
    )


class _MethaneBackend:
    projection_convention = "pure_spin_highest_ms"
    consumes_uhf = True
    method_supported = True

    @property
    def identity(self):
        from hotpot.cheminfo.adapters.calculation.contracts import BackendIdentity
        return BackendIdentity("test:explicit-methane:all-electron")

    @property
    def scientific_operation_requirements(self):
        from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import ScientificOperationRequirements, ScientificQuantity
        return ScientificOperationRequirements(
            "explicit-methane-projection",
            frozenset({ScientificQuantity.NET_CHARGE, ScientificQuantity.ELECTRONIC_STATE}),
            frozenset(),
        )

    @staticmethod
    def supports_atomic_number(atomic_number):
        return atomic_number in {1, 6}

    @staticmethod
    def effective_nuclear_charge(atomic_number):
        return atomic_number


@pytest.fixture(scope="module")
def methane_workflow(methane_facts):
    from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import resolve_and_project_state
    from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

    return resolve_and_project_state(methane_facts, _MethaneBackend(), StateRequest(net_charge=0, two_s=0))


def test_explicit_methane_preserves_charge_electrons_and_spin(methane_workflow):
    bundle = methane_workflow.scientific_state
    assert tuple(candidate.charge for result in bundle.net_charge_results for candidate in result.candidates) == (0,)
    assert tuple(
        (candidate.electron_count, candidate.two_s, candidate.multiplicity)
        for result in bundle.electronic_state_results for candidate in result.candidates
    ) == ((10, 0, 1),)
    projection = methane_workflow.backend_projection
    assert projection.status.value == "projected"
    assert (projection.projection.charge, projection.projection.effective_electron_count, projection.projection.uhf, projection.projection.multiplicity) == (0, 10, 0, 1)


def test_odd_spin_for_even_electron_methane_never_gets_projected(methane_facts):
    from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import resolve_and_project_state
    from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

    workflow = resolve_and_project_state(methane_facts, _MethaneBackend(), StateRequest(net_charge=0, two_s=1))
    assert workflow.backend_projection.projection is None
    assert workflow.backend_projection.status.value == "not_projected"
    assert any(
        finding.code == "HARD-006.PARITY_MISMATCH"
        for result in workflow.scientific_state.electronic_state_results
        for finding in result.findings
    )


def test_optional_profile_is_still_planned_and_ineligible(methane_facts):
    from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
    from hotpot.cheminfo.operations.calculation.scientific_state.profiles import _coordination_screening_profile
    from hotpot.cheminfo.operations.calculation.scientific_state.state_resolver import resolve_scientific_state

    profile = _coordination_screening_profile()
    assert profile.activation_status == "planned"
    assert profile.content_digest == _RESOURCE_DIGESTS["coordination_screening_v1.json"]
    bundle = resolve_scientific_state(methane_facts, policy=StatePolicy.COORDINATION_SCREENING_V1)
    eligibility = bundle.execution_decision.static_profile_eligibility
    assert eligibility.in_domain is False
    assert "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE" in {finding.code for finding in eligibility.findings}


def test_current_pickle_cold_loads_the_new_owners_and_keeps_values(methane_workflow, tmp_path):
    from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

    bundle = methane_workflow.scientific_state
    values = {
        "facts": bundle.facts,
        "request": StateRequest(net_charge=0, two_s=0),
        "charge_result": bundle.net_charge_results[0],
        "electronic_result": bundle.electronic_state_results[0],
        "decision": bundle.execution_decision,
        "projection": methane_workflow.backend_projection,
        "bundle": bundle,
        "workflow": methane_workflow,
    }
    expected_owners = {
        "facts": "hotpot.cheminfo.structure.snapshots",
        "request": _RULES + ".request",
        "charge_result": _RULES + ".charge.contracts",
        "electronic_result": _RULES + ".electronic_state.contracts",
        "decision": _OPERATIONS + ".state_selection",
        "projection": _OPERATIONS + ".execution_selection",
        "bundle": _OPERATIONS + ".state_resolver",
        "workflow": _OPERATIONS + ".execution_selection",
    }
    assert {name: type(value).__module__ for name, value in values.items()} == expected_owners
    source = tmp_path / "current-state.pickle"
    restored = tmp_path / "cold-state.pickle"
    source.write_bytes(pickle.dumps((values, expected_owners), protocol=pickle.HIGHEST_PROTOCOL))
    script = """
import pickle
from pathlib import Path
import sys
assert not any(name.startswith('hotpot') for name in sys.modules)
values, expected_owners = pickle.loads(Path(sys.argv[1]).read_bytes())
assert {name: type(value).__module__ for name, value in values.items()} == expected_owners
for name in ("charge_result", "electronic_result"):
    assert tuple(vars(values[name])) == (
        "identity", "facts_identity", "candidates",
        "admissibility", "resolution", "findings",
    )
assert not any(name.startswith(('hotpot.cheminfo.calculator.state_', 'hotpot.cheminfo.calculator.charge', 'hotpot.cheminfo.calculator.electronic_state', 'hotpot.cheminfo.calculator.oxidation_state', 'hotpot.cheminfo.calculator.contracts', 'hotpot.cheminfo.calculator.execution_selection')) for name in sys.modules)
Path(sys.argv[2]).write_bytes(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL))
"""
    result = subprocess.run([sys.executable, "-c", script, str(source), str(restored)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert pickle.loads(restored.read_bytes()) == values
