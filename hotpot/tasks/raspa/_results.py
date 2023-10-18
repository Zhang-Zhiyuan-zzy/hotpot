"""
python v3.9.0
@Project: hotpot
@File   : _results
@Auther : Zhiyuan Zhang
@Data   : 2023/9/26:q

@Time   : 9:42
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd


class RaspaParser:
    """split long output to filter useful information"""
    current_cycle_pattern = r"Current cycle: (\d+) out of (\d+)"
    net_charge_pattern = r"Net charge:\s+([\d.]+)"
    F_charge_pattern = r"F:\s+([\d.]+)"
    A_charge_pattern = r"A:\s+([\d.]+)"
    C_charge_pattern = r"C:\s+([\d.]+)"
    current_box_pattern = r"Current Box:(.*?)Average Box:"
    average_box_pattern = r"Average Box:(.*?)Box-lengths:"
    current_box_pattern_start = r"Current Box:\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*\[A\]\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*\[A\]\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*\[A\]"
    box_lengths_pattern = r"Box-lengths:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    box_lengths_avg_pattern = r"Box-lengths:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[A\]\s+Average:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[A\]"
    box_angles_pattern = r"Box-angles:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    box_angles_avg_pattern = r"Box-angles:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[degrees\]\s+Average:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[degrees\]"
    volume_pattern = r"Volume:\s+([\d.]+)"
    avg_volume_pattern = r"Average Volume:\s+([\d.]+)"
    component_split_pattern = r"Component (\d+) \((\w+)\), current number of integer/fractional/reaction molecules: (\d+)/(\d+)/(\d+) \(avg\.\s+([\d.]+)\), density:\s+([\d.]+) \(avg\.\s+([\d.]+)\) \[kg/m\^3\]\s+absolute adsorption:\s+([\d.]+) \(avg\.\s+([\d.]+)\) \[mol/uc\],\s+([\d.]+) \(avg\.\s+([\d.]+)\) \[mol/kg\],\s+([\d.]+) \(avg\.\s+([\d.]+)\) \[mg/g\].+?([\d.]+) \(avg\.\s+([\d.]+)\) \[cm\^3 STP/g\],\s+([\d.]+) \(avg\.\s+([\d.]+)\) \[cm\^3 STP/cm\^3\]\s+excess adsorption:\s*(-?[\d.]+) \(avg.\s*(-?[\d.]+)\) \[mol/uc\],\s*(-?[\d.]+) \(avg.\s*(-?[\d.]+)\) \[mol/kg\],\s*(-?[\d.]+) \(avg.\s*(-?[\d.]+)\) \[mg/g\]\s*(-?[\d.]+) \(avg.\s*(-?[\d.]+)\) \[cm\^3 STP/g\],\s*(-?[\d.]+) \(avg.\s*(-?[\d.]+)\) \[cm\^3 STP/cm\^3\]"
    component_split_pattern_start = r"Component (\d+) \((\w+)\), current number of integer/fractional/reaction molecules: (\d+)/(\d+)/(\d+), density:\s+([\d.]+) \[kg/m\^3\].+?absolute adsorption:\s+([\d.-]+) \[mol/uc\],\s+([\d.-]+) \[mol/kg\],\s+([\d.-]+) \[mg/g\].+?([\d.-]+) \[cm\^3 STP/g\],\s+([\d.-]+) \[cm\^3 STP/cm\^3\].+?excess adsorption:\s+([\d.-]+) \[mol/uc\],\s+([\d.-]+) \[mol/kg\],\s+([\d.-]+) \[mg/g\].+?([\d.-]+) \[cm\^3 STP/g\],\s+([\d.-]+) \[cm\^3 STP/cm\^3\]"
    Degrees_of_freedom_pattern = r"Degrees of freedom:\s*(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)"
    framework_atoms = r"Number of Framework-atoms:\s*(-?[\d.]+)"
    Adsorbates_num_pattern = r"Number of Adsorbates:\s*(-?[\d.]+)"  #
    cations_num_pattern = r"Number of Cations:\s*(-?[\d.]+)"
    ctp_energy_pattern = r"Current total potential energy:\s*(-?[\d.]+)"
    chh_energy_pattern = r"Current Host-Host energy:\s*(-?[\d.]+)"
    cha_energy_pattern = r"Current Host-Adsorbate energy:\s*(-?[\d.]+)"
    chc_energy_pattern = r"Current Host-Cation energy:\s*(-?[\d.]+)"
    caa_energy_pattern = r"Current Adsorbate-Adsorbate energy:\s*(-?[\d.]+)"
    ccc_energy_pattern = r"Current Cation-Cation energy:\s*(-?[\d.]+)"
    cac_energy_pattern = r"Current Adsorbate-Cation energy:\s*(-?[\d.]+)"
    numeric_pattern = r"(-?[\d.]+)"
    ctp_energy_avg_pattern = r"Current total potential energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    chh_energy_avg_pattern = r"Current Host-Host energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    cha_energy_avg_pattern = r"Current Host-Adsorbate energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    chc_energy_avg_pattern = r"Current Host-Cation energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    caa_energy_avg_pattern = r"Current Adsorbate-Adsorbate energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    ccc_energy_avg_pattern = r"Current Cation-Cation energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    cac_energy_avg_pattern = r"Current Adsorbate-Cation energy:\s*" + numeric_pattern + r"\s*\[K\]\s*\(avg.\s*" + numeric_pattern + r"\)"
    average_properties_current_cycle_pattern = r"Average Properties at Current cycle: (\d+) out of (\d+)"
    framework_surface_area_pattern = r"Framework surface area:\s+([\d.]+)\s+\[m\^2/g\]\s+([\d.]+)\s+\[m\^2/cm\^3\]\s+([\d.]+)\s+\[A\^2\]"
    Framework0_individual_surface_area_pattern = r"Framework 0 individual surface area:\s+([\d.]+)\s+\[m\^2/g\]\s+([\d.]+)\s+\[m\^2/cm\^3\]\s+([\d.]+)\s+\[A\^2\]"
    cation_surface_area = r"Cation surface area:\s+([\d.]+)\s+\[m\^2/g\]\s+([\d.]+)\s+\[m\^2/cm\^3\]\s+([\d.]+)\s+\[A\^2\]"
    compressibility_pattern = r"Compressibility:\s+(-?[\d.]+|\S+)"
    henry_pattern = r"Component (\d+): (\d+) \[mol/kg/Pa\]"

    def __init__(self, output=None, _data: np.ndarray = None):
        if isinstance(_data, np.ndarray):
            self.data = _data
        else:
            self.text = output
            self.patterns = {
                'start': re.compile(r"Starting simulation"),
                'split': re.compile(r"Average Properties at Current cycle: (\d+) out of (\d+)"),
                'end': re.compile(r"Finishing simulation"),
            }
            self._dict = None
            self.data = self.parse()

    def __repr__(self):
        """"""
        return f"RaspaResult:\n{self.data}"

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            return self.__class__(_data=self.data[np.array(keys)])
        else:
            return self.__class__(_data=self.data[keys])

    def process_text(self):
        start_match = self.patterns['start'].search(self.text)
        split_matches = list(self.patterns['split'].finditer(self.text))
        end_match = self.patterns['end'].search(self.text)

        if start_match and end_match:
            start_index = start_match.start()
            end_index = end_match.end()

            blocks = []
            block1 = self.text[start_index:split_matches[0].start()].strip()
            blocks.append(block1)
            for i in range(len(split_matches) - 1):
                start_idx = split_matches[i].start()
                end_idx = split_matches[i + 1].start()
                block = self.text[start_idx:end_idx].strip()
                blocks.append(block)

            block0 = self.text[split_matches[-1].start():end_index].strip()
            blocks.append(block0)
            return blocks

    def parse(self) -> np.ndarray:
        """extract useful information from text into dictionary"""
        result_dict = {'blocks': []}

        for block_index, block in enumerate(self.process_text(), start=1):
            block_result = {'block': f'{block_index}'}

            average_properties_match = re.search(self.average_properties_current_cycle_pattern, block)
            if average_properties_match:
                block_result['average properties current cycle'] = f"{average_properties_match.group(1)} out " \
                                                                   f"of {average_properties_match.group(2)}"
            else:
                block_result['average properties current cycle'] = 'initialization'

            block_result['framework surface area'] = None
            framework_surface_area_match = re.search(self.framework_surface_area_pattern, block)
            if framework_surface_area_match:
                values = [
                    float(framework_surface_area_match.group(1)),
                    float(framework_surface_area_match.group(2)),
                    float(framework_surface_area_match.group(3)),
                ]

                block_result['framework surface area'] = ','.join(map(str, values))

            block_result['framework 0 individual surface area'] = None
            framework0_surface_area_match = re.search(self.Framework0_individual_surface_area_pattern, block)
            if framework0_surface_area_match:
                values = [
                    float(framework0_surface_area_match.group(1)),
                    float(framework0_surface_area_match.group(2)),
                    float(framework0_surface_area_match.group(3)),
                ]

                block_result['framework 0 individual surface area'] = ','.join(map(str, values))

            block_result['cation surface area'] = None
            cation_surface_area_match = re.search(self.cation_surface_area, block)
            if framework0_surface_area_match:
                values = [
                    float(cation_surface_area_match.group(1)),
                    float(cation_surface_area_match.group(2)),
                    float(cation_surface_area_match.group(3)),
                ]

                block_result['cation surface area'] = ','.join(map(str, values))

            block_result['Compressibility'] = None
            compressibility_match = re.search(self.compressibility_pattern, block)
            if compressibility_match:
                block_result['Compressibility'] = compressibility_match.group(1)

            block_result['henry_coefficients'] = None
            henry_pattern_matches = re.finditer(self.henry_pattern, block)
            henry_coefficients = []

            for match in henry_pattern_matches:
                value = int(match.group(2))
                henry_coefficients.append(str(value))

            if henry_coefficients:
                coefficients_str = ', '.join(henry_coefficients)
                block_result['henry_coefficients'] = coefficients_str

            current_cycle_match = re.search(self.current_cycle_pattern, block)
            if current_cycle_match:
                if block_index == 1:
                    block_result['current_cycle'] = 'initialization'
                elif block_index != 1:
                    block_result['current_cycle'] = f"{current_cycle_match.group(1)} out of {current_cycle_match.group(2)}"

            block_result['charge'] = {
                'net_charge': None,
                'F': None,
                'A': None,
                'C': None,
            }

            net_charge_match = re.search(self.net_charge_pattern, block)
            if net_charge_match:
                block_result['charge']['net_charge'] = float(net_charge_match.group(1))

            F_charge_match = re.search(self.F_charge_pattern, block)
            if F_charge_match:
                block_result['charge']['F'] = float(F_charge_match.group(1))

            A_charge_match = re.search(self.A_charge_pattern, block)
            if A_charge_match:
                block_result['charge']['A'] = float(A_charge_match.group(1))

            C_charge_match = re.search(self.C_charge_pattern, block)
            if C_charge_match:
                block_result['charge']['C'] = float(C_charge_match.group(1))


            block_result['box'] = {
                'current': None,
                'average': None,
            }
            current_box_match = re.search(self.current_box_pattern, block, re.DOTALL | re.MULTILINE)
            average_box_match = re.search(self.average_box_pattern, block, re.DOTALL | re.MULTILINE)
            if current_box_match and average_box_match:
                current_box_values = current_box_match.group(1).strip().split()
                average_box_values = average_box_match.group(1).strip().split()
                block_result[
                    'box']['current'] = f"{current_box_values[0]} {current_box_values[1]} {current_box_values[2]} " \
                                     f"{average_box_values[4]} {average_box_values[5]} {average_box_values[6]} " \
                                     f"{average_box_values[12]} {average_box_values[13]} {average_box_values[14]} "
                block_result[
                    'box']['average'] = f"{average_box_values[0]} {average_box_values[1]} {average_box_values[2]} " \
                                     f"{average_box_values[8]} {average_box_values[9]} {average_box_values[10]} " \
                                     f"{average_box_values[16]} {average_box_values[17]} {average_box_values[18]} "

            current_box_match_start = re.search(self.current_box_pattern_start, block)

            if current_box_match_start:
                block_result[
                    'box']['current'] = f"{current_box_match_start.group(1)} {current_box_match_start.group(2)} {current_box_match_start.group(3)} " \
                                     f"{current_box_match_start.group(4)} {current_box_match_start.group(5)} {current_box_match_start.group(6)} " \
                                     f"{current_box_match_start.group(7)} {current_box_match_start.group(8)} {current_box_match_start.group(9)} "


            block_result['box_lengths'] = {
                'current': None,
                'average': None,
            }
            box_lengths_match = re.search(self.box_lengths_pattern, block)
            box_lengths_avg_match = re.search(self.box_lengths_avg_pattern, block)

            if box_lengths_match:
                block_result[
                    'box_lengths']['current'] = f"{box_lengths_match.group(1)} {box_lengths_match.group(2)} {box_lengths_match.group(3)}"

            if box_lengths_avg_match and box_lengths_match:
                block_result[
                    'box_lengths'][
                    'average'] = f"{box_lengths_avg_match.group(1)} {box_lengths_avg_match.group(2)} {box_lengths_avg_match.group(3)}"
                block_result[
                    'box_lengths'][
                    'current'] = f"{box_lengths_match.group(1)} {box_lengths_match.group(2)} {box_lengths_match.group(3)}"

            block_result['box_angles'] = {
                'current': None,
                'average': None,
            }
            box_angles_match = re.search(self.box_angles_pattern, block)
            if box_angles_match:
                block_result[
                    'box_angles']['current'] = f"{box_angles_match.group(1)} {box_angles_match.group(2)} {box_angles_match.group(3)}"

            box_angles_avg_match = re.search(self.box_angles_avg_pattern, block)
            if box_angles_avg_match and box_angles_match:
                block_result[
                    'box_angles'][
                    'average'] = f"{box_angles_avg_match.group(1)} {box_angles_avg_match.group(2)} {box_angles_avg_match.group(3)}"
                block_result[
                    'box_angles'][
                    'current'] = f"{box_angles_match.group(1)} {box_angles_match.group(2)} {box_angles_match.group(3)}"

            block_result['volume'] = {
                'current': None,
                'average': None,
            }
            volume_match = re.search(self.volume_pattern, block)
            if volume_match:
                block_result['volume']['current'] = float(volume_match.group(1))

            avg_volume_match = re.search(self.avg_volume_pattern, block)
            if avg_volume_match and volume_match:
                block_result['volume']['average'] = float(avg_volume_match.group(1))
                block_result['volume']['current'] = float(volume_match.group(1))

            component_matches_start = re.compile(self.component_split_pattern_start, re.DOTALL).finditer(block)
            for match in component_matches_start:
                # component_number = match.group(1)
                component_name = match.group(2)

                component_info = {
                    'density': float(match.group(6)),
                    'number_of_molecules': {
                        'integer_molecules': int(match.group(3)),
                        'fractional_molecules': float(match.group(4)),
                        'reaction_molecules': int(match.group(5)),
                    },
                    'absolute_adsorption': {
                        'uc': float(match.group(7)),
                        'kg': float(match.group(8)),
                        'mg': float(match.group(9)),
                        'cm3_stp_g': float(match.group(10)),
                        'cm3_stp_cm3': float(match.group(11)),
                        'avg_uc': None,
                        'avg_kg': None,
                        'avg_mg': None,
                        'avg_cm3_stp_g': None,
                        'avg_cm3_stp_cm3': None,
                    },
                    'excess_adsorption': {
                        'uc': float(match.group(12)),
                        'kg': float(match.group(13)),
                        'mg': float(match.group(14)),
                        'cm3_stp_g': float(match.group(15)),
                        'cm3_stp_cm3': float(match.group(16)),
                        'avg_uc': None,
                        'avg_kg': None,
                        'avg_mg': None,
                        'avg_cm3_stp_g': None,
                        'avg_cm3_stp_cm3': None,
                    }
                }

                block_result[f'{component_name}'] = component_info


            component_list = []
            component_matches = re.compile(self.component_split_pattern, re.DOTALL).finditer(block)
            for match in component_matches:
                component_name = match.group(2)

                component_list.append(component_name)

                integer_molecules = match.group(3)
                fractional_molecules = match.group(4)
                reaction_molecules = match.group(5)
                avg_molecules = match.group(6)
                density = match.group(7)
                avg_density = match.group(8)
                absolute_adsorption_uc = match.group(9)
                absolute_adsorption_uc_avg = match.group(10)
                absolute_adsorption_kg = match.group(11)
                absolute_adsorption_kg_avg= match.group(12)
                absolute_adsorption_mg = match.group(13)
                absolute_adsorption_mg_avg = match.group(14)
                absolute_adsorption_g = match.group(15)
                absolute_adsorption_g_avg = match.group(16)
                absolute_adsorption_cm3 = match.group(17)
                absolute_adsorption_cm3_avg = match.group(18)
                excess_adsorption_uc = match.group(19)
                excess_adsorption_uc_avg = match.group(20)
                excess_adsorption_kg = match.group(21)
                excess_adsorption_kg_avg = match.group(22)
                excess_adsorption_mg = match.group(23)
                excess_adsorption_mg_avg = match.group(24)
                excess_adsorption_g = match.group(25)
                excess_adsorption_g_avg = match.group(26)
                excess_adsorption_cm3 = match.group(27)
                excess_adsorption_cm3_avg = match.group(28)

                component_info = {
                    'density': density,
                    'avg_density': avg_density,
                    'number_of_molecules': {
                        'integer_molecules': integer_molecules,
                        'fractional_molecules': fractional_molecules,
                        'reaction_molecules': reaction_molecules,
                        'average_molecules': avg_molecules,
                },

                    'absolute_adsorption': {
                        'uc': absolute_adsorption_uc,
                        'kg': absolute_adsorption_kg,
                        'mg': absolute_adsorption_mg,
                        'cm3_stp_g': absolute_adsorption_g,
                        'cm3_stp_cm3': absolute_adsorption_cm3,
                        'avg_uc': absolute_adsorption_uc_avg,
                        'avg_kg': absolute_adsorption_kg_avg,
                        'avg_mg': absolute_adsorption_mg_avg,
                        'avg_cm3_stp_g': absolute_adsorption_g_avg,
                        'avg_cm3_stp_cm3': absolute_adsorption_cm3_avg,
                },
                    'excess_adsorption': {
                        'uc': excess_adsorption_uc,
                        'kg': excess_adsorption_kg,
                        'mg': excess_adsorption_mg,
                        'cm3_stp_g': excess_adsorption_g,
                        'cm3_stp_cm3': excess_adsorption_cm3.strip(),
                        'avg_uc': excess_adsorption_uc_avg,
                        'avg_kg': excess_adsorption_kg_avg,
                        'avg_mg': excess_adsorption_mg_avg,
                        'avg_cm3_stp_g': excess_adsorption_g_avg,
                        'avg_cm3_stp_cm3': excess_adsorption_cm3_avg.strip(),
                    }
                }

                block_result[f'{component_name}'] = component_info


            degrees_of_freedom_match = re.search(self.Degrees_of_freedom_pattern, block)
            if degrees_of_freedom_match:
                values = [
                    float(degrees_of_freedom_match.group(1)),
                    float(degrees_of_freedom_match.group(2)),
                    float(degrees_of_freedom_match.group(3)),
                    float(degrees_of_freedom_match.group(4)),
                ]

                block_result['degrees_of_freedom'] = ','.join(map(str, values))

            framework_atoms_match = re.search(self.framework_atoms, block)
            if framework_atoms_match:
                block_result['Number of framework_atoms'] = float(framework_atoms_match.group(1))

            adsorbates_num_match = re.search(self.Adsorbates_num_pattern, block)
            if adsorbates_num_match:
                block_result['Number of Adsorbates'] = float(adsorbates_num_match.group(1))

            cations_num_match = re.search(self.cations_num_pattern, block)
            if cations_num_match:
                block_result['Number of Cations'] = float(cations_num_match.group(1))


            block_result['total potential energy'] = {
                'current': None,
                'average': None,
            }
            ctp_energy_match = re.search(self.ctp_energy_pattern, block)
            if ctp_energy_match:
                block_result['total potential energy']['current'] = float(ctp_energy_match.group(1))

            ctp_energy_avg_match = re.search(self.ctp_energy_avg_pattern, block)
            if ctp_energy_avg_match and ctp_energy_match:
                block_result['total potential energy']['average'] = float(ctp_energy_avg_match.group(2))
                block_result['total potential energy']['current'] = float(ctp_energy_match.group(1))


            block_result['Host-Host energy'] = {
                'current': None,
                'average': None,
            }
            chh_energy_match = re.search(self.chh_energy_pattern, block)
            if chh_energy_match:
                block_result['Host-Host energy']['current'] = float(chh_energy_match.group(1))
            chh_energy_avg_match = re.search(self.chh_energy_avg_pattern, block)

            if chh_energy_avg_match and chh_energy_match:
                block_result['Host-Host energy']['average'] = float(chh_energy_avg_match.group(2))
                block_result['Host-Host energy']['current'] = float(chh_energy_avg_match.group(1))


            block_result['Host-Adsorbate energy'] = {
                'current': None,
                'average': None,
            }
            cha_energy_match = re.search(self.cha_energy_pattern, block)
            cha_energy_avg_match = re.search(self.cha_energy_avg_pattern, block)
            if cha_energy_match:
                block_result['Host-Adsorbate energy']['current'] = float(cha_energy_match.group(1))

            if cha_energy_avg_match and cha_energy_match:
                block_result['Host-Adsorbate energy']['average'] = float(cha_energy_avg_match.group(2))
                block_result['Host-Adsorbate energy']['current'] = float(cha_energy_avg_match.group(1))



            block_result['Host-Cation energy'] = {
                'current': None,
                'average': None,
            }
            chc_energy_match = re.search(self.chc_energy_pattern, block)
            if chc_energy_match:
                block_result['Host-Cation energy']['current'] = float(chc_energy_match.group(1))
            chc_energy_avg_match = re.search(self.chc_energy_avg_pattern, block)
            if chc_energy_avg_match and chc_energy_match:
                block_result['Host-Cation energy']['average'] = float(chc_energy_avg_match.group(2))
                block_result['Host-Cation energy']['current'] = float(chc_energy_match.group(1))



            block_result['Adsorbate-Adsorbate energy'] = {
                'current': None,
                'average': None,
            }
            caa_energy_match = re.search(self.caa_energy_pattern, block)
            if caa_energy_match:
                block_result['Adsorbate-Adsorbate energy']['current'] = float(caa_energy_match.group(1))
            caa_energy_avg_match = re.search(self.caa_energy_avg_pattern, block)
            if caa_energy_avg_match and caa_energy_match:
                block_result['Adsorbate-Adsorbate energy']['average'] = float(caa_energy_avg_match.group(2))
                block_result['Adsorbate-Adsorbate energy']['current'] = float(caa_energy_match.group(1))



            block_result['Cation-Cation energy'] = {
                'current': None,
                'average': None,
            }
            ccc_energy_match = re.search(self.ccc_energy_pattern, block)
            if ccc_energy_match:
                block_result['Cation-Cation energy']['current'] = float(ccc_energy_match.group(1))
            ccc_energy_avg_match = re.search(self.ccc_energy_avg_pattern, block)
            if ccc_energy_avg_match:
                block_result['Cation-Cation energy']['average'] = float(ccc_energy_avg_match.group(2))
                block_result['Cation-Cation energy']['current'] = float(ccc_energy_match.group(1))

            block_result['Adsorbate-Cation energy'] = {
                'current': None,
                'average': None,
            }
            cac_energy_match = re.search(self.cac_energy_pattern, block)
            if cac_energy_match:
                block_result['Adsorbate-Cation energy']['current'] = float(cac_energy_match.group(1))
            cac_energy_avg_match = re.search(self.cac_energy_avg_pattern, block)
            if cac_energy_avg_match and cac_energy_match:
                block_result['Adsorbate-Cation energy']['average'] = float(cac_energy_avg_match.group(2))
                block_result['Adsorbate-Cation energy']['current'] = float(cac_energy_match.group(1))


            result_dict['blocks'].append(block_result)
            self._dict = result_dict
            component_dtypes = []
            for component_name in component_list:
                component_dtype = (
                    f'{component_name}',
                    [
                        ('density', float),
                        ('avg_density', float),

                         ('number_of_molecules',
                          [('integer_molecules', float), ('fractional_molecules', float), ('reaction_molecules', float),
                          ('average_molecules', float)]),

                         ('absolute_adsorption',
                           [('uc', float), ('kg', float), ('mg', float), ('cm3_stp_g', float), ('cm3_stp_cm3', float),
                            ('avg_uc', float), ('avg_kg', float), ('avg_mg', float), ('avg_cm3_stp_g', float), ('avg_cm3_stp_cm3', float)]),

                         ('excess_adsorption',
                          [('uc', float), ('kg', float), ('mg', float), ('cm3_stp_g', float), ('cm3_stp_cm3', float),
                           ('avg_uc', float), ('avg_kg', float), ('avg_mg', float), ('avg_cm3_stp_g', float), ('avg_cm3_stp_cm3', float)])
                    ]
                )

                component_dtypes.append(component_dtype)

            dtypes = [
                ('block', float),
                ('average properties current cycle', '<U20'),
                ('framework surface area', '<U20'),
                ('framework 0 individual surface area', '<U20'),
                ('cation surface area', '<U20'),
                ('Compressibility', '<U20'),
                ('henry_coefficients', '<U20'),
                ('current_cycle', '<U20'),
                ('charge', [('net_charge', float),
                            ('F', float),
                            ('A', float),
                            ('C', float)]),
                ('box', [('current', '<U80'),
                         ('average', '<U80')]),
                ('box_lengths', [('current', '<U20'),
                                 ('average', '<U20')]),
                ('box_angles', [('current', '<U20'),
                                ('average', '<U20')]),
                ('volume', [('current', float),
                            ('average', float)]),
                ('degrees_of_freedom', '<U20'),
                ('Number of framework_atoms', float),
                ('Number of Adsorbates', float),
                ('Number of Cations', float),
                ('total potential energy', [('current', float), ('average', float)]),
                ('Host-Host energy', [('current', float), ('average', float)]),
                ('Host-Adsorbate energy', [('current', float), ('average', float)]),
                ('Host-Cation energy', [('current', float), ('average', float)]),
                ('Adsorbate-Adsorbate energy', [('current', float), ('average', float)]),
                ('Cation-Cation energy', [('current', float), ('average', float)]),
                ('Adsorbate-Cation energy', [('current', float), ('average', float)])
            ]

            for i in component_dtypes:
                dtypes.append(i)


            blocks_array = np.empty(len(result_dict['blocks']), dtype=dtypes)
            for i, block_data in enumerate(result_dict['blocks']):
                component_data = []
                for component_name in component_list:
                    component_data.extend([
                        (block_data.get(component_name, {}).get('density', np.nan),
                         block_data.get(component_name, {}).get('avg_density', np.nan),
                        (
                            block_data.get(component_name, {}).get('number_of_molecules', {}).get('integer_molecules',
                                                                                                  np.nan),
                            block_data.get(component_name, {}).get('number_of_molecules', {}).get(
                                'fractional_molecules', np.nan),
                            block_data.get(component_name, {}).get('number_of_molecules', {}).get('reaction_molecules',
                                                                                                  np.nan),
                            block_data.get(component_name, {}).get('number_of_molecules', {}).get('average_molecules',
                                                                                                  np.nan)
                        ),
                        (
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('uc', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('kg', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('mg', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('cm3_stp_g', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('cm3_stp_cm3',
                                                                                                  np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('avg_uc', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('avg_kg', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('avg_mg', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('avg_cm3_stp_g', np.nan),
                            block_data.get(component_name, {}).get('absolute_adsorption', {}).get('avg_cm3_stp_cm3',
                                                                                                  np.nan)
                        ),
                        (
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('uc', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('kg', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('mg', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('cm3_stp_g', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('cm3_stp_cm3', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('avg_uc', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('avg_kg', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('avg_mg', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('avg_cm3_stp_g', np.nan),
                            block_data.get(component_name, {}).get('excess_adsorption', {}).get('avg_cm3_stp_cm3', np.nan),
                          )
                        )
                    ])
            for i, block_data in enumerate(result_dict['blocks']):
                blocks_array[i] = (
                    block_data['block'],
                    block_data['average properties current cycle'],
                    block_data.get('framework surface area', np.nan),
                    block_data.get('framework 0 individual surface area', np.nan),
                    block_data.get('cation surface area', np.nan),
                    block_data.get('Compressibility', np.nan),
                    block_data.get('henry_coefficients', np.nan),
                    block_data['current_cycle'],
                    (block_data['charge']['net_charge'],
                     block_data['charge']['F'],
                     block_data['charge']['A'],
                     block_data['charge']['C']),
                    (block_data['box']['current'],
                     block_data['box']['average']),
                    (block_data['box_lengths']['current'],
                     block_data['box_lengths']['average']),
                    (block_data['box_angles']['current'],
                     block_data['box_angles']['average']),
                    (block_data['volume']['current'],
                     block_data['volume']['average']),
                    block_data['degrees_of_freedom'],
                    block_data['Number of framework_atoms'],
                    block_data['Number of Adsorbates'],
                    block_data['Number of Cations'],
                    (block_data['total potential energy']['current'],
                     block_data['total potential energy']['average']),
                    (block_data['Host-Host energy']['current'],
                     block_data['Host-Host energy']['average']),
                    (block_data['Host-Adsorbate energy']['current'],
                     block_data['Host-Adsorbate energy']['average']),
                    (block_data['Host-Cation energy']['current'],
                     block_data['Host-Cation energy']['average']),
                    (block_data['Adsorbate-Adsorbate energy']['current'],
                     block_data['Adsorbate-Adsorbate energy']['average']),
                    (block_data['Cation-Cation energy']['current'],
                     block_data['Cation-Cation energy']['average']),
                    (block_data['Adsorbate-Cation energy']['current'],
                     block_data['Adsorbate-Cation energy']['average']),
                    *component_data)


        return blocks_array



    def keys(self):
        try:
            return self._get_keys_recursive(self.data.dtype),
        except:
            return []



    def _get_keys_recursive(self, dtype, prefix=''):
        keys = []
        for field_name in dtype.names:
            keys.append(f'{prefix}{field_name}')
        return keys

    @staticmethod
    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(RaspaParser.flatten(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(RaspaParser.flatten({f"{new_key}{sep}{i}_{k}": val for k, val in item.items()}, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)



    def excel_output(self, output_dir: str):
        dict_output = self._dict
        result_list = dict_output['blocks']
        flattened_result = [RaspaParser.flatten(item) for item in result_list]
        dataframe = pd.DataFrame(flattened_result)
        dataframe.to_excel(output_dir, index=False)

