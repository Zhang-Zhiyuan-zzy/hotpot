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

text = """Compiler and run-time data
        ===========================================================================
        RASPA 2.0.47
        Compiled as a 64-bits application
        Compiler: gcc 11.4.0
        Compile Date = Sep 25 2023, Compile Time = 06:57:46

        Tue Sep 26 01:25:24 2023
        Simulation started on Tuesday, September 26.
        The start time was 01:25 AM.

        Hostname:    3090
        OS type:     Linux on x86_64
        OS release:  5.15.0-83-generic
        OS version:  #92-Ubuntu SMP Mon Aug 14 09:30:42 UTC 2023

        Simulation
        ===========================================================================
        Dimensions: 3
        Random number seed: 1695691524
        RASPA directory set to: /home/pub/RASPA/simulations
        String appended to output-files: 
        Number of cycles: 100000
        Number of initializing cycles: 10000
        Number of equilibration cycles: 0
        Print every: 10000
        Triclinic boundary condition applied
        Timestep: 0.000500
            Degrees of freedom:                        0
            Translational Degrees of freedom:          0
            Rotational Degrees of freedom:             0
            Degrees of freedom Framework:              0


        Mutual consistent basic set of units:
        ======================================
        Unit of temperature: Kelvin
        Unit of length:      1e-10 [m]
        Unit of time:        1e-12 [s]
        Unit of mass:        1.66054e-27 [kg]
        Unit of charge:      1.60218e-19 [C/particle]

        Derived units and their conversion factors:
        ===========================================
        Unit of energy:              1.66054e-23 [J]
        Unit of force:               1.66054e-13 [N]
        Unit of pressure:            1.66054e+07 [Pa]
        Unit of velocity:            100 [m/s]
        Unit of acceleration:        1e-08 [m^2/s]
        Unit of diffusion:           1e-08 [m^2/s]
        Unit of dipole moment:       1.60218e-29 [C.m]
        Unit of electric potential:  0.000103643 [V]
        Unit of electric field:      1.03643e+06 [V]
        Unit of polarizability:      1.54587e-35 [-]
        Unit of Coulomb potential:   167101.0800066561  [K]
        Unit of dielectric constant: 0.0000154587       [s^2 C^2/(kg m^3)]
        Unit of wave vectors:        5.3088374589       [cm^1]
        Boltzmann constant:          0.8314464919       [-]

        Internal conversion factors:
        ===========================================
        Energy to Kelvin:                                    1.2027242847
        FH correction factor                                 2.0211930949
        Heat capacity conversion factor:                    10.0000088723
        From Debye to internal units:                        4.8032067991
        Isothermal compressibility conversion factor:        0.0000000602

        Energy conversion factors:
        ===========================================
        From mdyne/A to kcal/mol/A^2:           143.933
        From mdyne/A to kj/mol/A^2:             602.214
        From mdyne/A to K/A^2:                  72429.7
        From mdyne A/rad^2 to kcal/mol/deg^2:   0.0438444


        Properties computed
        ===========================================================================
        Movies: no
        Radial Distribution Function: no
        Number of molecules GCMC histogram: no
        Histogram of the molecule positions: no
        Free energy profiles: no
        Pore Size Distribution Function: no
        End-to-end distance: no
        Histogram of the energy of the system: no
        Compute thermodynamic factors: no
        Framework spacing histograms: no
        Residence times histograms: no
        Distance histograms: no
        Bend Angle histograms: no
        Dihedral angle histograms: no
        Angle between planes histograms: no
        Molecule properties: no
        Infra-red spectra: no
        Mean-squared displacement using modified order-N algorithm: no
        Velocity-autocorrelation function modified order-N algorithm: no
        Rotational velocity-autocorrelation function modified order-N algorithm: no
        Molecular orientation-autocorrelation function modified order-N algorithm: no
        Bond orientation-autocorrelation function modified order-N algorithm: no
        Mean-squared displacement (conventional algorithm): no
        Velocity-autocorrelation function (conventional algorithm): no
        3D density grid for adsorbates: no
        Compute cation an/or adsorption sites: no
        dcTST snapshots: no
        Compute pressure and stress: no


        VTK
        ===========================================================================
        VTK fractional-range position framework atoms: [-0.001000,1.001000] [-0.001000,1.001000] [-0.001000,1.001000]
        VTK fractional-range position framework bonds: [-0.151000,1.151000] [-0.151000,1.151000] [-0.151000,1.151000]
        VTK fractional-range com-position adsorbate molecules: [-0.101000,1.101000] [-0.101000,1.101000] [-0.101000,1.101000]
        VTK fractional-range com-position cation molecules: [-0.101000,1.101000] [-0.101000,1.101000] [-0.101000,1.101000]
            3D free energy grid made for the full simulation-cell


        Thermo/Baro-stat NHC parameters
        ===========================================================================
        External temperature: 273.15 [K]
        Beta: 0.00440316 [energy unit]
        External Pressure: 101325 [Pa]


        Thermostat chain-length: 3
        Timescale parameter for thermostat: 0.150000 [ps]
        Barostat chain-length:   3
        Timescale parameter for barostat:   0.150000 [ps]

        Number of Yoshida-Suzuki decomposition steps: 5
        Number of respa steps: 5


        Method and settings for electrostatics
        ===============================================================================
        Dielectric constant of the medium : 1.000000
        Charge from charge-equilibration: no
        Ewald summation is used (exact solution of a periodic system)
        Relative precision                : 1e-06
        Alpha convergence parameter       : 0.265058
        kvec (x,y,z)                      : 7 7 7


        CFC-RXMC parameters
        ===========================================================================
        Number of reactions: 0


        Rattle parameters
        ===========================================================================
        Distance constraint type: r^2-r^2_0
        Bend angle constraint type: theta-theta_0
        Dihedral angle constraint type: phi-phi_0
        Inversion-bend angle constraint type: chi-chi_0
        Out-of-plane distance constraint type: r-r_0


        Spectra parameters
        ===========================================================================
        Compute normal modes: no


        Minimization parameters
        ===========================================================================
        Generalized coordinates are: Cartesian center-of-mass, elements of the orientational matrix p1,p2,p3 and strain
        Potential derivatives are evaluated: analytically
        Translation of the system is removed from the generalized Hessian: no
        Rotation of the system is removed from the generalized Hessian: no
        Maximum step-length: 0.3
        Convergence factor: 1
        Maximum number of minimization steps: 10000
        Use gradients in the line-minimizations: yes
        RMS gradient tolerance: 1e-06
        Maximum gradient tolerance: 1e-06

        Distance constraints: 0
        Angle constraints: 0
        Dihedral constraints: 0

        Improper dihedral constraints: 0

        Inversion-bend constraints: 0

        Out-of-plane constraints: 0

        Harmonic distance constraints: 0
        Harmonic angle constraints: 0
        Harmonic dihedral constraints: 0

        Dihedral mid-point measurements: 0

        All framework atoms are fixed

        Fixed adsorbate atoms:  
        Fixed adsorbate groups (center-of-mass):  
        Fixed adsorbate groups (orientation):  

        Fixed cation atoms:  
        Fixed cation groups (center-of-mass):  
        Fixed cation groups (orientation):  


        dcTST parameters
        ===========================================================================
        Free energy profiles computed: no
        Free energy profiles written every 5000 cycles
        Free energy mapping: mapped to a,b,c-coordinates
        BarrierPosition:       0.0000000000       0.0000000000       0.0000000000
        BarrierNormal:         0.0000000000       0.0000000000       0.0000000000
        Start with a molecule on top of the barrier: no
        Maximum distance to barrier (e.g. distance to minumum free energy):       0.0000000000 [A]
        Maximum trajectory time:      10.0000000000 [ps]
        Each configuration is used with 5 different initial velocities


        Cbmc parameters
        ===========================================================================
        Biasing method: using only the VDW part
        Number of trial positions:                                       10
        Number of trial positions (reinsertion):                         10
        Number of trial positions (partial reinsertion):                 10
        Number of trial positions (identity-change):                     10
        Number of trial positions (Gibbs particle transfer):             10
        Number of trial positions (insertion/deletion):                  10
        Number of trial positions (Widom insertion):                     10
        Number of trial positions coupled Torsion-selection:             100
        Number of trial positions first bead:                            10
        Number of trial positions first bead (reinsertion):              10
        Number of trial positions first bead (partial reinsertion):      10
        Number of trial positions first bead (identity-change):          10
        Number of trial positions first bead (Gibbs particle transfer):  10
        Number of trial positions first bead (insertion/deletion):       10
        Number of trial positions first bead (Widom insertion):          10
        Number of trial moves per open bead:                             150
        Target acceptance ratio small-mc scheme:                         0.400000
        Energy overlap criteria:                                         1e+07
        Minimal Rosenbluth factor:                                       1e-150


        Pseudo atoms: 40
        ===========================================================================
        Pseudo Atom[   0] Name UNIT     Oxidation:          Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.000000000  B-factor:0.000   
                         Charge=1.000000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Absolute), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   1] Name He       Oxidation: +0       Element: He   pdb-name: He   Scat. Types:   3   2 Mass=4.002602000  B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   2] Name CH4_sp3  Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=16.042460000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   3] Name CH3_sp3  Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=15.034520000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   4] Name CH2_sp3  Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=14.026580000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   5] Name CH3_sp3_ethane Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=15.034520000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   6] Name CH2_sp2_ethene Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=14.026580000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   7] Name I2_united_atom Oxidation: +0       Element: I    pdb-name: I    Scat. Types:  54  49 Mass=253.808900000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   8] Name CH_sp3   Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=13.018640000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[   9] Name C_sp3    Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=12.000000000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  10] Name H_h2     Oxidation: +0       Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.007940000  B-factor:1.000   
                         Charge=0.468000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  11] Name H_com    Oxidation: +0       Element: H    pdb-name: H    Scat. Types:   1   1 Mass=0.000000000  B-factor:1.000   
                         Charge=-0.936000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  12] Name C_co2    Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=12.000000000 B-factor:1.000   
                         Charge=0.700000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.720 [A], Framework-atom:  no
        Pseudo Atom[  13] Name O_co2    Oxidation: +0       Element: O    pdb-name: O    Scat. Types:   9   8 Mass=15.999400000 B-factor:1.000   
                         Charge=-0.350000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.680 [A], Framework-atom:  no
        Pseudo Atom[  14] Name O_o2     Oxidation: +0       Element: O    pdb-name: O    Scat. Types:   9   8 Mass=15.999400000 B-factor:1.000   
                         Charge=-0.113000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  15] Name O_com    Oxidation: +0       Element: -    pdb-name: O    Scat. Types:   0   8 Mass=0.000000000  B-factor:1.000   
                         Charge=0.226000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  16] Name N_n2     Oxidation: +0       Element: N    pdb-name: N    Scat. Types:   8   7 Mass=14.006740000 B-factor:1.000   
                         Charge=-0.482000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  17] Name N_com    Oxidation: +0       Element: -    pdb-name: N    Scat. Types:   0   7 Mass=0.000000000  B-factor:1.000   
                         Charge=0.964000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  18] Name Ar       Oxidation: +0       Element: Ar   pdb-name: Ar   Scat. Types:  19  18 Mass=39.948000000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  19] Name Ow       Oxidation: +0       Element: O    pdb-name: O    Scat. Types:   9   8 Mass=15.999400000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.500 [A], Framework-atom:  no
        Pseudo Atom[  20] Name Hw       Oxidation: +0       Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.007940000  B-factor:1.000   
                         Charge=0.241000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  21] Name Lw       Oxidation: +0       Element: H    pdb-name: L    Scat. Types:   1   3 Mass=0.000000000  B-factor:1.000   
                         Charge=-0.241000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  22] Name C_benz   Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=12.000000000 B-factor:1.000   
                         Charge=-0.095000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.700 [A], Framework-atom:  no
        Pseudo Atom[  23] Name H_benz   Oxidation: +0       Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.007940000  B-factor:1.000   
                         Charge=0.095000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.320 [A], Framework-atom:  no
        Pseudo Atom[  24] Name N_dmf    Oxidation: +0       Element: N    pdb-name: N    Scat. Types:   8   7 Mass=14.006740000 B-factor:1.000   
                         Charge=-0.570000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.500 [A], Framework-atom:  no
        Pseudo Atom[  25] Name Co_dmf   Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=12.000000000 B-factor:1.000   
                         Charge=0.450000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.520 [A], Framework-atom:  no
        Pseudo Atom[  26] Name Cm_dmf   Oxidation: +0       Element: C    pdb-name: C    Scat. Types:   7   6 Mass=12.000000000 B-factor:1.000   
                         Charge=0.280000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.520 [A], Framework-atom:  no
        Pseudo Atom[  27] Name O_dmf    Oxidation: +0       Element: O    pdb-name: O    Scat. Types:   9   8 Mass=15.999400000 B-factor:1.000   
                         Charge=-0.500000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.780 [A], Framework-atom:  no
        Pseudo Atom[  28] Name H_dmf    Oxidation: +0       Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.007940000  B-factor:1.000   
                         Charge=0.060000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.220 [A], Framework-atom:  no
        Pseudo Atom[  29] Name Na       Oxidation: +0       Element: Na   pdb-name: Na   Scat. Types:  12  11 Mass=22.989770000 B-factor:1.000   
                         Charge=1.000000000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  30] Name Cl       Oxidation: +0       Element: Cl   pdb-name: Cl   Scat. Types:  18  17 Mass=35.453000000 B-factor:1.000   
                         Charge=-1.000000000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  31] Name Kr       Oxidation: +0       Element: Kr   pdb-name: Kr   Scat. Types:  37  36 Mass=83.798000000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  32] Name Xe       Oxidation: +0       Element: Xe   pdb-name: Xe   Scat. Types:  55  54 Mass=131.293000000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  33] Name O_tip4p_2005 Oxidation: +0       Element: O    pdb-name: O    Scat. Types:   9   8 Mass=15.999400000 B-factor:1.000   
                         Charge=0.000000000          Polarization=0.000000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    0.500 [A], Framework-atom:  no
        Pseudo Atom[  34] Name H_tip4p_2005 Oxidation: +0       Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.008000000  B-factor:1.000   
                         Charge=0.556400000          Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  35] Name M_tip4p_2005 Oxidation: +0       Element: -    pdb-name: O    Scat. Types:   0   8 Mass=0.000000000  B-factor:1.000   
                         Charge=-1.112800000         Polarization=0.000000000  [A^3] (considered a charged atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Relative), Radius:    1.000 [A], Framework-atom:  no
        Pseudo Atom[  36] Name Mof_Zn   Oxidation:          Element: Zn   pdb-name: Zn   Scat. Types:  31  30 Mass=65.408728088 B-factor:0.000   
                         Charge=0.000000000  (-na-)  Polarization=5.750000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Absolute), Radius:    1.200 [A], Framework-atom: yes (charge definition not found)
        Pseudo Atom[  37] Name Mof_O    Oxidation:          Element: O    pdb-name: O    Scat. Types:   9   8 Mass=15.999404927 B-factor:0.000   
                         Charge=0.000000000  (-na-)  Polarization=0.802000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Absolute), Radius:    0.640 [A], Framework-atom: yes (charge definition not found)
        Pseudo Atom[  38] Name Mof_C    Oxidation:          Element: C    pdb-name: C    Scat. Types:   7   6 Mass=12.010735897 B-factor:0.000   
                         Charge=0.000000000  (-na-)  Polarization=1.760000000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Absolute), Radius:    0.750 [A], Framework-atom: yes (charge definition not found)
        Pseudo Atom[  39] Name Mof_H    Oxidation:          Element: H    pdb-name: H    Scat. Types:   1   1 Mass=1.007940754  B-factor:0.000   
                         Charge=0.000000000  (-na-)  Polarization=0.666793000  [A^3] (considered a chargeless atom and no polarization)  Interactions: yes
                         Anisotropic factor:    0.000 [-] (Absolute), Radius:    0.320 [A], Framework-atom: yes (charge definition not found)


        Forcefield: UFF
        ===========================================================================
        Minimal distance: 1
        CutOff VDW : 12.800000 (163.840000)
        CutOff VDW switching on: 11.520000 (132.710400)
        CutOff charge-charge : 12.000000 (144.000000)
        CutOff charge-charge switching on: 7.800000 (60.840000)
        CutOff charge-bonddipole : 12.000000 (144.000000)
        CutOff charge-bondipole switching on: 8.400000 (70.560000)
        CutOff bonddipole-bonddipole : 12.000000 (144.000000)
        CutOff bonddipole-bondipole switching on: 9.000000 (81.000000)
        Polarization is neglected
        TailCorrections are used
        All potentials are shifted to zero at the Cutoff

        General mixing rule: Lorentz-Berthelot mixing rules are used FIRST for cross terms
        0 cross terms are overwritten using the individual mixing rules from the file 'force_field_mixing_rules.def'
        and then 0 terms are overwritten using the specific interactions from the file 'force_field.def'

        The force field and all the interactions:
             He -      He [LENNARD_JONES] p_0/k_B:  10.90000 [K], p_1: 2.64000 [A], shift/k_B:  -0.00335596 [K], tailcorrection: yes
             He - CH4_sp3 [LENNARD_JONES] p_0/k_B:  40.16466 [K], p_1: 3.18500 [A], shift/k_B:  -0.03812394 [K], tailcorrection: yes
             He - CH3_sp3 [LENNARD_JONES] p_0/k_B:  32.68333 [K], p_1: 3.19500 [A], shift/k_B:  -0.03161161 [K], tailcorrection: yes
             He - CH2_sp3 [LENNARD_JONES] p_0/k_B:  22.39196 [K], p_1: 3.29500 [A], shift/k_B:  -0.02605544 [K], tailcorrection: yes
             He - CH3_sp3_ethane [LENNARD_JONES] p_0/k_B:  32.68333 [K], p_1: 3.19500 [A], shift/k_B:  -0.03161161 [K], tailcorrection: yes
             He - CH2_sp2_ethene [LENNARD_JONES] p_0/k_B:  30.43846 [K], p_1: 3.15750 [A], shift/k_B:  -0.02742743 [K], tailcorrection: yes
             He - I2_united_atom [LENNARD_JONES] p_0/k_B:  77.42739 [K], p_1: 3.81100 [A], shift/k_B:  -0.21558862 [K], tailcorrection: yes
             He -  CH_sp3 [LENNARD_JONES] p_0/k_B:  13.61249 [K], p_1: 3.65500 [A], shift/k_B:  -0.02950032 [K], tailcorrection: yes
             He -   C_sp3 [LENNARD_JONES] p_0/k_B:   2.95296 [K], p_1: 4.51000 [A], shift/k_B:  -0.02255724 [K], tailcorrection: yes
             He -    H_h2 [ZERO_POTENTIAL]
             He -   H_com [LENNARD_JONES] p_0/k_B:  19.30751 [K], p_1: 2.80000 [A], shift/k_B:  -0.00846110 [K], tailcorrection: yes
             He -   C_co2 [LENNARD_JONES] p_0/k_B:  17.15517 [K], p_1: 2.72000 [A], shift/k_B:  -0.00631784 [K], tailcorrection: yes
             He -   O_co2 [LENNARD_JONES] p_0/k_B:  29.34451 [K], p_1: 2.84500 [A], shift/k_B:  -0.01415038 [K], tailcorrection: yes
             He -    O_o2 [LENNARD_JONES] p_0/k_B:  23.11060 [K], p_1: 2.83000 [A], shift/k_B:  -0.01079641 [K], tailcorrection: yes
             He -   O_com [ZERO_POTENTIAL]
             He -    N_n2 [LENNARD_JONES] p_0/k_B:  19.80909 [K], p_1: 2.97500 [A], shift/k_B:  -0.01248872 [K], tailcorrection: yes
             He -   N_com [ZERO_POTENTIAL]
             He -      Ar [LENNARD_JONES] p_0/k_B:  36.77893 [K], p_1: 3.03000 [A], shift/k_B:  -0.02588086 [K], tailcorrection: yes
             He -      Ow [LENNARD_JONES] p_0/k_B:  31.25699 [K], p_1: 2.86850 [A], shift/k_B:  -0.01583513 [K], tailcorrection: yes
             He -      Hw [LENNARD_JONES] p_0/k_B:  15.53467 [K], p_1: 2.60500 [A], shift/k_B:  -0.00441487 [K], tailcorrection: yes
             He -  C_benz [LENNARD_JONES] p_0/k_B:  18.29289 [K], p_1: 3.12000 [A], shift/k_B:  -0.01534331 [K], tailcorrection: yes
             He -  H_benz [LENNARD_JONES] p_0/k_B:  16.65548 [K], p_1: 2.50000 [A], shift/k_B:  -0.00369805 [K], tailcorrection: yes
             He -   N_dmf [LENNARD_JONES] p_0/k_B:  29.52965 [K], p_1: 2.92000 [A], shift/k_B:  -0.01664540 [K], tailcorrection: yes
             He -  Co_dmf [LENNARD_JONES] p_0/k_B:  23.34524 [K], p_1: 3.17000 [A], shift/k_B:  -0.02154040 [K], tailcorrection: yes
             He -  Cm_dmf [LENNARD_JONES] p_0/k_B:  29.52965 [K], p_1: 3.22000 [A], shift/k_B:  -0.02992840 [K], tailcorrection: yes
             He -   O_dmf [LENNARD_JONES] p_0/k_B:  33.01515 [K], p_1: 2.80000 [A], shift/k_B:  -0.01446818 [K], tailcorrection: yes
             He -   H_dmf [LENNARD_JONES] p_0/k_B:   9.33809 [K], p_1: 2.42000 [A], shift/k_B:  -0.00170581 [K], tailcorrection: yes
             He -      Na [LENNARD_JONES] p_0/k_B:  12.82501 [K], p_1: 2.65000 [A], shift/k_B:  -0.00403923 [K], tailcorrection: yes
             He -      Cl [LENNARD_JONES] p_0/k_B:  35.28298 [K], p_1: 3.08000 [A], shift/k_B:  -0.02738959 [K], tailcorrection: yes
             He -      Kr [LENNARD_JONES] p_0/k_B:  42.43442 [K], p_1: 3.15000 [A], shift/k_B:  -0.03769513 [K], tailcorrection: yes
             He -      Xe [LENNARD_JONES] p_0/k_B:  50.04818 [K], p_1: 3.30500 [A], shift/k_B:  -0.05930462 [K], tailcorrection: yes
             He - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  31.87287 [K], p_1: 2.89950 [A], shift/k_B:  -0.01722271 [K], tailcorrection: yes
             He - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.32000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             He - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.32000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             He -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  17.52604 [K], p_1: 2.68000 [A], shift/k_B:  -0.00590549 [K], tailcorrection: yes
             He -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  17.52604 [K], p_1: 2.68000 [A], shift/k_B:  -0.00590549 [K], tailcorrection: yes
             He -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  17.52604 [K], p_1: 2.68000 [A], shift/k_B:  -0.00590549 [K], tailcorrection: yes
             He -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  17.52604 [K], p_1: 2.68000 [A], shift/k_B:  -0.00590549 [K], tailcorrection: yes
        CH4_sp3 - CH4_sp3 [LENNARD_JONES] p_0/k_B: 148.00000 [K], p_1: 3.73000 [A], shift/k_B:  -0.36228376 [K], tailcorrection: yes
        CH4_sp3 - CH3_sp3 [LENNARD_JONES] p_0/k_B: 120.43255 [K], p_1: 3.74000 [A], shift/k_B:  -0.29957347 [K], tailcorrection: yes
        CH4_sp3 - CH2_sp3 [LENNARD_JONES] p_0/k_B:  82.51061 [K], p_1: 3.84000 [A], shift/k_B:  -0.24042553 [K], tailcorrection: yes
        CH4_sp3 - CH3_sp3_ethane [LENNARD_JONES] p_0/k_B: 120.43255 [K], p_1: 3.74000 [A], shift/k_B:  -0.29957347 [K], tailcorrection: yes
        CH4_sp3 - CH2_sp2_ethene [LENNARD_JONES] p_0/k_B: 112.16060 [K], p_1: 3.70250 [A], shift/k_B:  -0.26263732 [K], tailcorrection: yes
        CH4_sp3 - I2_united_atom [LENNARD_JONES] p_0/k_B: 285.30685 [K], p_1: 4.35600 [A], shift/k_B:  -1.76996415 [K], tailcorrection: yes
        CH4_sp3 -  CH_sp3 [LENNARD_JONES] p_0/k_B:  50.15974 [K], p_1: 4.20000 [A], shift/k_B:  -0.25009722 [K], tailcorrection: yes
        CH4_sp3 -   C_sp3 [LENNARD_JONES] p_0/k_B:  10.88118 [K], p_1: 5.05500 [A], shift/k_B:  -0.16449483 [K], tailcorrection: yes
        CH4_sp3 -    H_h2 [ZERO_POTENTIAL]
        CH4_sp3 -   H_com [LENNARD_JONES] p_0/k_B:  71.14492 [K], p_1: 3.34500 [A], shift/k_B:  -0.09061132 [K], tailcorrection: yes
        CH4_sp3 -   C_co2 [LENNARD_JONES] p_0/k_B:  63.21392 [K], p_1: 3.26500 [A], shift/k_B:  -0.06962936 [K], tailcorrection: yes
        CH4_sp3 -   O_co2 [LENNARD_JONES] p_0/k_B: 108.12955 [K], p_1: 3.39000 [A], shift/k_B:  -0.14920826 [K], tailcorrection: yes
        CH4_sp3 -    O_o2 [LENNARD_JONES] p_0/k_B:  85.15868 [K], p_1: 3.37500 [A], shift/k_B:  -0.11442628 [K], tailcorrection: yes
        CH4_sp3 -   O_com [ZERO_POTENTIAL]
        CH4_sp3 -    N_n2 [LENNARD_JONES] p_0/k_B:  72.99315 [K], p_1: 3.52000 [A], shift/k_B:  -0.12622645 [K], tailcorrection: yes
        CH4_sp3 -   N_com [ZERO_POTENTIAL]
        CH4_sp3 -      Ar [LENNARD_JONES] p_0/k_B: 135.52417 [K], p_1: 3.57500 [A], shift/k_B:  -0.25719765 [K], tailcorrection: yes
        CH4_sp3 -      Ow [LENNARD_JONES] p_0/k_B: 115.17675 [K], p_1: 3.41350 [A], shift/k_B:  -0.16565639 [K], tailcorrection: yes
        CH4_sp3 -      Hw [LENNARD_JONES] p_0/k_B:  57.24264 [K], p_1: 3.15000 [A], shift/k_B:  -0.05084950 [K], tailcorrection: yes
        CH4_sp3 -  C_benz [LENNARD_JONES] p_0/k_B:  67.40623 [K], p_1: 3.66500 [A], shift/k_B:  -0.14849257 [K], tailcorrection: yes
        CH4_sp3 -  H_benz [LENNARD_JONES] p_0/k_B:  61.37263 [K], p_1: 3.04500 [A], shift/k_B:  -0.04448565 [K], tailcorrection: yes
        CH4_sp3 -   N_dmf [LENNARD_JONES] p_0/k_B: 108.81176 [K], p_1: 3.46500 [A], shift/k_B:  -0.17120819 [K], tailcorrection: yes
        CH4_sp3 -  Co_dmf [LENNARD_JONES] p_0/k_B:  86.02325 [K], p_1: 3.71500 [A], shift/k_B:  -0.20554612 [K], tailcorrection: yes
        CH4_sp3 -  Cm_dmf [LENNARD_JONES] p_0/k_B: 108.81176 [K], p_1: 3.76500 [A], shift/k_B:  -0.28169851 [K], tailcorrection: yes
        CH4_sp3 -   O_dmf [LENNARD_JONES] p_0/k_B: 121.65525 [K], p_1: 3.34500 [A], shift/k_B:  -0.15494209 [K], tailcorrection: yes
        CH4_sp3 -   H_dmf [LENNARD_JONES] p_0/k_B:  34.40930 [K], p_1: 2.96500 [A], shift/k_B:  -0.02125970 [K], tailcorrection: yes
        CH4_sp3 -      Na [LENNARD_JONES] p_0/k_B:  47.25802 [K], p_1: 3.19500 [A], shift/k_B:  -0.04570837 [K], tailcorrection: yes
        CH4_sp3 -      Cl [LENNARD_JONES] p_0/k_B: 130.01185 [K], p_1: 3.62500 [A], shift/k_B:  -0.26816805 [K], tailcorrection: yes
        CH4_sp3 -      Kr [LENNARD_JONES] p_0/k_B: 156.36368 [K], p_1: 3.69500 [A], shift/k_B:  -0.36171900 [K], tailcorrection: yes
        CH4_sp3 -      Xe [LENNARD_JONES] p_0/k_B: 184.41909 [K], p_1: 3.85000 [A], shift/k_B:  -0.54581909 [K], tailcorrection: yes
        CH4_sp3 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B: 117.44616 [K], p_1: 3.44450 [A], shift/k_B:  -0.17833277 [K], tailcorrection: yes
        CH4_sp3 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.86500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH4_sp3 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.86500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH4_sp3 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  64.58049 [K], p_1: 3.22500 [A], shift/k_B:  -0.06606459 [K], tailcorrection: yes
        CH4_sp3 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  64.58049 [K], p_1: 3.22500 [A], shift/k_B:  -0.06606459 [K], tailcorrection: yes
        CH4_sp3 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  64.58049 [K], p_1: 3.22500 [A], shift/k_B:  -0.06606459 [K], tailcorrection: yes
        CH4_sp3 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  64.58049 [K], p_1: 3.22500 [A], shift/k_B:  -0.06606459 [K], tailcorrection: yes
        CH3_sp3 - CH3_sp3 [LENNARD_JONES] p_0/k_B:  98.00000 [K], p_1: 3.75000 [A], shift/k_B:  -0.24770750 [K], tailcorrection: yes
        CH3_sp3 - CH2_sp3 [LENNARD_JONES] p_0/k_B:  67.14164 [K], p_1: 3.85000 [A], shift/k_B:  -0.19871690 [K], tailcorrection: yes
        CH3_sp3 - CH3_sp3_ethane [LENNARD_JONES] p_0/k_B:  98.00000 [K], p_1: 3.75000 [A], shift/k_B:  -0.24770750 [K], tailcorrection: yes
        CH3_sp3 - CH2_sp2_ethene [LENNARD_JONES] p_0/k_B:  91.26883 [K], p_1: 3.71250 [A], shift/k_B:  -0.21720151 [K], tailcorrection: yes
        CH3_sp3 - I2_united_atom [LENNARD_JONES] p_0/k_B: 232.16374 [K], p_1: 4.36600 [A], shift/k_B:  -1.46020036 [K], tailcorrection: yes
        CH3_sp3 -  CH_sp3 [LENNARD_JONES] p_0/k_B:  40.81666 [K], p_1: 4.21000 [A], shift/k_B:  -0.20643346 [K], tailcorrection: yes
        CH3_sp3 -   C_sp3 [LENNARD_JONES] p_0/k_B:   8.85438 [K], p_1: 5.06500 [A], shift/k_B:  -0.13544546 [K], tailcorrection: yes
        CH3_sp3 -    H_h2 [ZERO_POTENTIAL]
        CH3_sp3 -   H_com [LENNARD_JONES] p_0/k_B:  57.89300 [K], p_1: 3.35500 [A], shift/k_B:  -0.07506553 [K], tailcorrection: yes
        CH3_sp3 -   C_co2 [LENNARD_JONES] p_0/k_B:  51.43928 [K], p_1: 3.27500 [A], shift/k_B:  -0.05770867 [K], tailcorrection: yes
        CH3_sp3 -   O_co2 [LENNARD_JONES] p_0/k_B:  87.98864 [K], p_1: 3.40000 [A], shift/k_B:  -0.12357986 [K], tailcorrection: yes
        CH3_sp3 -    O_o2 [LENNARD_JONES] p_0/k_B:  69.29646 [K], p_1: 3.38500 [A], shift/k_B:  -0.09477957 [K], tailcorrection: yes
        CH3_sp3 -   O_com [ZERO_POTENTIAL]
        CH3_sp3 -    N_n2 [LENNARD_JONES] p_0/k_B:  59.39697 [K], p_1: 3.53000 [A], shift/k_B:  -0.10447722 [K], tailcorrection: yes
        CH3_sp3 -   N_com [ZERO_POTENTIAL]
        CH3_sp3 -      Ar [LENNARD_JONES] p_0/k_B: 110.28055 [K], p_1: 3.58500 [A], shift/k_B:  -0.21282585 [K], tailcorrection: yes
        CH3_sp3 -      Ow [LENNARD_JONES] p_0/k_B:  93.72318 [K], p_1: 3.42350 [A], shift/k_B:  -0.13718612 [K], tailcorrection: yes
        CH3_sp3 -      Hw [LENNARD_JONES] p_0/k_B:  46.58025 [K], p_1: 3.16000 [A], shift/k_B:  -0.04217219 [K], tailcorrection: yes
        CH3_sp3 -  C_benz [LENNARD_JONES] p_0/k_B:  54.85071 [K], p_1: 3.67500 [A], shift/k_B:  -0.12282397 [K], tailcorrection: yes
        CH3_sp3 -  H_benz [LENNARD_JONES] p_0/k_B:  49.94097 [K], p_1: 3.05500 [A], shift/k_B:  -0.03691851 [K], tailcorrection: yes
        CH3_sp3 -   N_dmf [LENNARD_JONES] p_0/k_B:  88.54377 [K], p_1: 3.47500 [A], shift/k_B:  -0.14174676 [K], tailcorrection: yes
        CH3_sp3 -  Co_dmf [LENNARD_JONES] p_0/k_B:  70.00000 [K], p_1: 3.72500 [A], shift/k_B:  -0.16997772 [K], tailcorrection: yes
        CH3_sp3 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  88.54377 [K], p_1: 3.77500 [A], shift/k_B:  -0.23290246 [K], tailcorrection: yes
        CH3_sp3 -   O_dmf [LENNARD_JONES] p_0/k_B:  98.99495 [K], p_1: 3.35500 [A], shift/k_B:  -0.12835935 [K], tailcorrection: yes
        CH3_sp3 -   H_dmf [LENNARD_JONES] p_0/k_B:  28.00000 [K], p_1: 2.97500 [A], shift/k_B:  -0.01765272 [K], tailcorrection: yes
        CH3_sp3 -      Na [LENNARD_JONES] p_0/k_B:  38.45543 [K], p_1: 3.20500 [A], shift/k_B:  -0.03789824 [K], tailcorrection: yes
        CH3_sp3 -      Cl [LENNARD_JONES] p_0/k_B: 105.79499 [K], p_1: 3.63500 [A], shift/k_B:  -0.22185229 [K], tailcorrection: yes
        CH3_sp3 -      Kr [LENNARD_JONES] p_0/k_B: 127.23836 [K], p_1: 3.70500 [A], shift/k_B:  -0.29915207 [K], tailcorrection: yes
        CH3_sp3 -      Xe [LENNARD_JONES] p_0/k_B: 150.06798 [K], p_1: 3.86000 [A], shift/k_B:  -0.45111296 [K], tailcorrection: yes
        CH3_sp3 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  95.56987 [K], p_1: 3.45450 [A], shift/k_B:  -0.14766056 [K], tailcorrection: yes
        CH3_sp3 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.87500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH3_sp3 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.87500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH3_sp3 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH3_sp3 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH3_sp3 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH3_sp3 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH2_sp3 - CH2_sp3 [LENNARD_JONES] p_0/k_B:  46.00000 [K], p_1: 3.95000 [A], shift/k_B:  -0.15876887 [K], tailcorrection: yes
        CH2_sp3 - CH3_sp3_ethane [LENNARD_JONES] p_0/k_B:  67.14164 [K], p_1: 3.85000 [A], shift/k_B:  -0.19871690 [K], tailcorrection: yes
        CH2_sp3 - CH2_sp2_ethene [LENNARD_JONES] p_0/k_B:  62.52999 [K], p_1: 3.81250 [A], shift/k_B:  -0.17451965 [K], tailcorrection: yes
        CH2_sp3 - I2_united_atom [LENNARD_JONES] p_0/k_B: 159.05974 [K], p_1: 4.46600 [A], shift/k_B:  -1.14574651 [K], tailcorrection: yes
        CH2_sp3 -  CH_sp3 [LENNARD_JONES] p_0/k_B:  27.96426 [K], p_1: 4.31000 [A], shift/k_B:  -0.16279225 [K], tailcorrection: yes
        CH2_sp3 -   C_sp3 [LENNARD_JONES] p_0/k_B:   6.06630 [K], p_1: 5.16500 [A], shift/k_B:  -0.10429590 [K], tailcorrection: yes
        CH2_sp3 -    H_h2 [ZERO_POTENTIAL]
        CH2_sp3 -   H_com [LENNARD_JONES] p_0/k_B:  39.66359 [K], p_1: 3.45500 [A], shift/k_B:  -0.06133558 [K], tailcorrection: yes
        CH2_sp3 -   C_co2 [LENNARD_JONES] p_0/k_B:  35.24202 [K], p_1: 3.37500 [A], shift/k_B:  -0.04735411 [K], tailcorrection: yes
        CH2_sp3 -   O_co2 [LENNARD_JONES] p_0/k_B:  60.28267 [K], p_1: 3.50000 [A], shift/k_B:  -0.10074403 [K], tailcorrection: yes
        CH2_sp3 -    O_o2 [LENNARD_JONES] p_0/k_B:  47.47631 [K], p_1: 3.48500 [A], shift/k_B:  -0.07732446 [K], tailcorrection: yes
        CH2_sp3 -   O_com [ZERO_POTENTIAL]
        CH2_sp3 -    N_n2 [LENNARD_JONES] p_0/k_B:  40.69398 [K], p_1: 3.63000 [A], shift/k_B:  -0.08463385 [K], tailcorrection: yes
        CH2_sp3 -   N_com [ZERO_POTENTIAL]
        CH2_sp3 -      Ar [LENNARD_JONES] p_0/k_B:  75.55528 [K], p_1: 3.68500 [A], shift/k_B:  -0.17196601 [K], tailcorrection: yes
        CH2_sp3 -      Ow [LENNARD_JONES] p_0/k_B:  64.21151 [K], p_1: 3.52350 [A], shift/k_B:  -0.11170425 [K], tailcorrection: yes
        CH2_sp3 -      Hw [LENNARD_JONES] p_0/k_B:  31.91301 [K], p_1: 3.26000 [A], shift/k_B:  -0.03483012 [K], tailcorrection: yes
        CH2_sp3 -  C_benz [LENNARD_JONES] p_0/k_B:  37.57925 [K], p_1: 3.77500 [A], shift/k_B:  -0.09884715 [K], tailcorrection: yes
        CH2_sp3 -  H_benz [LENNARD_JONES] p_0/k_B:  34.21549 [K], p_1: 3.15500 [A], shift/k_B:  -0.03068469 [K], tailcorrection: yes
        CH2_sp3 -   N_dmf [LENNARD_JONES] p_0/k_B:  60.66300 [K], p_1: 3.57500 [A], shift/k_B:  -0.11512620 [K], tailcorrection: yes
        CH2_sp3 -  Co_dmf [LENNARD_JONES] p_0/k_B:  47.95832 [K], p_1: 3.82500 [A], shift/k_B:  -0.13650337 [K], tailcorrection: yes
        CH2_sp3 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  60.66300 [K], p_1: 3.87500 [A], shift/k_B:  -0.18664651 [K], tailcorrection: yes
        CH2_sp3 -   O_dmf [LENNARD_JONES] p_0/k_B:  67.82330 [K], p_1: 3.45500 [A], shift/k_B:  -0.10488162 [K], tailcorrection: yes
        CH2_sp3 -   H_dmf [LENNARD_JONES] p_0/k_B:  19.18333 [K], p_1: 3.07500 [A], shift/k_B:  -0.01474726 [K], tailcorrection: yes
        CH2_sp3 -      Na [LENNARD_JONES] p_0/k_B:  26.34654 [K], p_1: 3.30500 [A], shift/k_B:  -0.03121934 [K], tailcorrection: yes
        CH2_sp3 -      Cl [LENNARD_JONES] p_0/k_B:  72.48214 [K], p_1: 3.73500 [A], shift/k_B:  -0.17885728 [K], tailcorrection: yes
        CH2_sp3 -      Kr [LENNARD_JONES] p_0/k_B:  87.17339 [K], p_1: 3.80500 [A], shift/k_B:  -0.24044309 [K], tailcorrection: yes
        CH2_sp3 -      Xe [LENNARD_JONES] p_0/k_B: 102.81440 [K], p_1: 3.96000 [A], shift/k_B:  -0.36028340 [K], tailcorrection: yes
        CH2_sp3 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  65.47671 [K], p_1: 3.55450 [A], shift/k_B:  -0.12004911 [K], tailcorrection: yes
        CH2_sp3 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.97500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH2_sp3 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.97500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH2_sp3 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  36.00389 [K], p_1: 3.33500 [A], shift/k_B:  -0.04503900 [K], tailcorrection: yes
        CH2_sp3 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  36.00389 [K], p_1: 3.33500 [A], shift/k_B:  -0.04503900 [K], tailcorrection: yes
        CH2_sp3 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  36.00389 [K], p_1: 3.33500 [A], shift/k_B:  -0.04503900 [K], tailcorrection: yes
        CH2_sp3 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  36.00389 [K], p_1: 3.33500 [A], shift/k_B:  -0.04503900 [K], tailcorrection: yes
        CH3_sp3_ethane - CH3_sp3_ethane [LENNARD_JONES] p_0/k_B:  98.00000 [K], p_1: 3.75000 [A], shift/k_B:  -0.24770750 [K], tailcorrection: yes
        CH3_sp3_ethane - CH2_sp2_ethene [LENNARD_JONES] p_0/k_B:  91.26883 [K], p_1: 3.71250 [A], shift/k_B:  -0.21720151 [K], tailcorrection: yes
        CH3_sp3_ethane - I2_united_atom [LENNARD_JONES] p_0/k_B: 232.16374 [K], p_1: 4.36600 [A], shift/k_B:  -1.46020036 [K], tailcorrection: yes
        CH3_sp3_ethane -  CH_sp3 [LENNARD_JONES] p_0/k_B:  40.81666 [K], p_1: 4.21000 [A], shift/k_B:  -0.20643346 [K], tailcorrection: yes
        CH3_sp3_ethane -   C_sp3 [LENNARD_JONES] p_0/k_B:   8.85438 [K], p_1: 5.06500 [A], shift/k_B:  -0.13544546 [K], tailcorrection: yes
        CH3_sp3_ethane -    H_h2 [ZERO_POTENTIAL]
        CH3_sp3_ethane -   H_com [LENNARD_JONES] p_0/k_B:  57.89300 [K], p_1: 3.35500 [A], shift/k_B:  -0.07506553 [K], tailcorrection: yes
        CH3_sp3_ethane -   C_co2 [LENNARD_JONES] p_0/k_B:  51.43928 [K], p_1: 3.27500 [A], shift/k_B:  -0.05770867 [K], tailcorrection: yes
        CH3_sp3_ethane -   O_co2 [LENNARD_JONES] p_0/k_B:  87.98864 [K], p_1: 3.40000 [A], shift/k_B:  -0.12357986 [K], tailcorrection: yes
        CH3_sp3_ethane -    O_o2 [LENNARD_JONES] p_0/k_B:  69.29646 [K], p_1: 3.38500 [A], shift/k_B:  -0.09477957 [K], tailcorrection: yes
        CH3_sp3_ethane -   O_com [ZERO_POTENTIAL]
        CH3_sp3_ethane -    N_n2 [LENNARD_JONES] p_0/k_B:  59.39697 [K], p_1: 3.53000 [A], shift/k_B:  -0.10447722 [K], tailcorrection: yes
        CH3_sp3_ethane -   N_com [ZERO_POTENTIAL]
        CH3_sp3_ethane -      Ar [LENNARD_JONES] p_0/k_B: 110.28055 [K], p_1: 3.58500 [A], shift/k_B:  -0.21282585 [K], tailcorrection: yes
        CH3_sp3_ethane -      Ow [LENNARD_JONES] p_0/k_B:  93.72318 [K], p_1: 3.42350 [A], shift/k_B:  -0.13718612 [K], tailcorrection: yes
        CH3_sp3_ethane -      Hw [LENNARD_JONES] p_0/k_B:  46.58025 [K], p_1: 3.16000 [A], shift/k_B:  -0.04217219 [K], tailcorrection: yes
        CH3_sp3_ethane -  C_benz [LENNARD_JONES] p_0/k_B:  54.85071 [K], p_1: 3.67500 [A], shift/k_B:  -0.12282397 [K], tailcorrection: yes
        CH3_sp3_ethane -  H_benz [LENNARD_JONES] p_0/k_B:  49.94097 [K], p_1: 3.05500 [A], shift/k_B:  -0.03691851 [K], tailcorrection: yes
        CH3_sp3_ethane -   N_dmf [LENNARD_JONES] p_0/k_B:  88.54377 [K], p_1: 3.47500 [A], shift/k_B:  -0.14174676 [K], tailcorrection: yes
        CH3_sp3_ethane -  Co_dmf [LENNARD_JONES] p_0/k_B:  70.00000 [K], p_1: 3.72500 [A], shift/k_B:  -0.16997772 [K], tailcorrection: yes
        CH3_sp3_ethane -  Cm_dmf [LENNARD_JONES] p_0/k_B:  88.54377 [K], p_1: 3.77500 [A], shift/k_B:  -0.23290246 [K], tailcorrection: yes
        CH3_sp3_ethane -   O_dmf [LENNARD_JONES] p_0/k_B:  98.99495 [K], p_1: 3.35500 [A], shift/k_B:  -0.12835935 [K], tailcorrection: yes
        CH3_sp3_ethane -   H_dmf [LENNARD_JONES] p_0/k_B:  28.00000 [K], p_1: 2.97500 [A], shift/k_B:  -0.01765272 [K], tailcorrection: yes
        CH3_sp3_ethane -      Na [LENNARD_JONES] p_0/k_B:  38.45543 [K], p_1: 3.20500 [A], shift/k_B:  -0.03789824 [K], tailcorrection: yes
        CH3_sp3_ethane -      Cl [LENNARD_JONES] p_0/k_B: 105.79499 [K], p_1: 3.63500 [A], shift/k_B:  -0.22185229 [K], tailcorrection: yes
        CH3_sp3_ethane -      Kr [LENNARD_JONES] p_0/k_B: 127.23836 [K], p_1: 3.70500 [A], shift/k_B:  -0.29915207 [K], tailcorrection: yes
        CH3_sp3_ethane -      Xe [LENNARD_JONES] p_0/k_B: 150.06798 [K], p_1: 3.86000 [A], shift/k_B:  -0.45111296 [K], tailcorrection: yes
        CH3_sp3_ethane - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  95.56987 [K], p_1: 3.45450 [A], shift/k_B:  -0.14766056 [K], tailcorrection: yes
        CH3_sp3_ethane - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.87500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH3_sp3_ethane - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.87500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH3_sp3_ethane -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH3_sp3_ethane -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH3_sp3_ethane -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH3_sp3_ethane -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  52.55131 [K], p_1: 3.23500 [A], shift/k_B:  -0.05476666 [K], tailcorrection: yes
        CH2_sp2_ethene - CH2_sp2_ethene [LENNARD_JONES] p_0/k_B:  85.00000 [K], p_1: 3.67500 [A], shift/k_B:  -0.19033552 [K], tailcorrection: yes
        CH2_sp2_ethene - I2_united_atom [LENNARD_JONES] p_0/k_B: 216.21748 [K], p_1: 4.32850 [A], shift/k_B:  -1.29141424 [K], tailcorrection: yes
        CH2_sp2_ethene -  CH_sp3 [LENNARD_JONES] p_0/k_B:  38.01316 [K], p_1: 4.17250 [A], shift/k_B:  -0.18221779 [K], tailcorrection: yes
        CH2_sp2_ethene -   C_sp3 [LENNARD_JONES] p_0/k_B:   8.24621 [K], p_1: 5.02750 [A], shift/k_B:  -0.12066175 [K], tailcorrection: yes
        CH2_sp2_ethene -    H_h2 [ZERO_POTENTIAL]
        CH2_sp2_ethene -   H_com [LENNARD_JONES] p_0/k_B:  53.91660 [K], p_1: 3.31750 [A], shift/k_B:  -0.06535166 [K], tailcorrection: yes
        CH2_sp2_ethene -   C_co2 [LENNARD_JONES] p_0/k_B:  47.90616 [K], p_1: 3.23750 [A], shift/k_B:  -0.05015757 [K], tailcorrection: yes
        CH2_sp2_ethene -   O_co2 [LENNARD_JONES] p_0/k_B:  81.94510 [K], p_1: 3.36250 [A], shift/k_B:  -0.10768475 [K], tailcorrection: yes
        CH2_sp2_ethene -    O_o2 [LENNARD_JONES] p_0/k_B:  64.53681 [K], p_1: 3.34750 [A], shift/k_B:  -0.08256428 [K], tailcorrection: yes
        CH2_sp2_ethene -   O_com [ZERO_POTENTIAL]
        CH2_sp2_ethene -    N_n2 [LENNARD_JONES] p_0/k_B:  55.31727 [K], p_1: 3.49250 [A], shift/k_B:  -0.09126413 [K], tailcorrection: yes
        CH2_sp2_ethene -   N_com [ZERO_POTENTIAL]
        CH2_sp2_ethene -      Ar [LENNARD_JONES] p_0/k_B: 102.70589 [K], p_1: 3.54750 [A], shift/k_B:  -0.18609431 [K], tailcorrection: yes
        CH2_sp2_ethene -      Ow [LENNARD_JONES] p_0/k_B:  87.28577 [K], p_1: 3.38600 [A], shift/k_B:  -0.11959595 [K], tailcorrection: yes
        CH2_sp2_ethene -      Hw [LENNARD_JONES] p_0/k_B:  43.38087 [K], p_1: 3.12250 [A], shift/k_B:  -0.03656130 [K], tailcorrection: yes
        CH2_sp2_ethene -  C_benz [LENNARD_JONES] p_0/k_B:  51.08327 [K], p_1: 3.63750 [A], shift/k_B:  -0.10756427 [K], tailcorrection: yes
        CH2_sp2_ethene -  H_benz [LENNARD_JONES] p_0/k_B:  46.51075 [K], p_1: 3.01750 [A], shift/k_B:  -0.03192733 [K], tailcorrection: yes
        CH2_sp2_ethene -   N_dmf [LENNARD_JONES] p_0/k_B:  82.46211 [K], p_1: 3.43750 [A], shift/k_B:  -0.12369380 [K], tailcorrection: yes
        CH2_sp2_ethene -  Co_dmf [LENNARD_JONES] p_0/k_B:  65.19202 [K], p_1: 3.68750 [A], shift/k_B:  -0.14898360 [K], tailcorrection: yes
        CH2_sp2_ethene -  Cm_dmf [LENNARD_JONES] p_0/k_B:  82.46211 [K], p_1: 3.73750 [A], shift/k_B:  -0.20430199 [K], tailcorrection: yes
        CH2_sp2_ethene -   O_dmf [LENNARD_JONES] p_0/k_B:  92.19544 [K], p_1: 3.31750 [A], shift/k_B:  -0.11174898 [K], tailcorrection: yes
        CH2_sp2_ethene -   H_dmf [LENNARD_JONES] p_0/k_B:  26.07681 [K], p_1: 2.93750 [A], shift/k_B:  -0.01523556 [K], tailcorrection: yes
        CH2_sp2_ethene -      Na [LENNARD_JONES] p_0/k_B:  35.81410 [K], p_1: 3.16750 [A], shift/k_B:  -0.03288927 [K], tailcorrection: yes
        CH2_sp2_ethene -      Cl [LENNARD_JONES] p_0/k_B:  98.52842 [K], p_1: 3.59750 [A], shift/k_B:  -0.19415671 [K], tailcorrection: yes
        CH2_sp2_ethene -      Kr [LENNARD_JONES] p_0/k_B: 118.49895 [K], p_1: 3.66750 [A], shift/k_B:  -0.26211690 [K], tailcorrection: yes
        CH2_sp2_ethene -      Xe [LENNARD_JONES] p_0/k_B: 139.76051 [K], p_1: 3.82250 [A], shift/k_B:  -0.39624287 [K], tailcorrection: yes
        CH2_sp2_ethene - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  89.00562 [K], p_1: 3.41700 [A], shift/k_B:  -0.12880427 [K], tailcorrection: yes
        CH2_sp2_ethene - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.83750 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH2_sp2_ethene - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.83750 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        CH2_sp2_ethene -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  48.94180 [K], p_1: 3.19750 [A], shift/k_B:  -0.04755957 [K], tailcorrection: yes
        CH2_sp2_ethene -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  48.94180 [K], p_1: 3.19750 [A], shift/k_B:  -0.04755957 [K], tailcorrection: yes
        CH2_sp2_ethene -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  48.94180 [K], p_1: 3.19750 [A], shift/k_B:  -0.04755957 [K], tailcorrection: yes
        CH2_sp2_ethene -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  48.94180 [K], p_1: 3.19750 [A], shift/k_B:  -0.04755957 [K], tailcorrection: yes
        I2_united_atom - I2_united_atom [LENNARD_JONES] p_0/k_B: 550.00000 [K], p_1: 4.98200 [A], shift/k_B:  -7.62206549 [K], tailcorrection: yes
        I2_united_atom -  CH_sp3 [LENNARD_JONES] p_0/k_B:  96.69540 [K], p_1: 4.82600 [A], shift/k_B:  -1.10784899 [K], tailcorrection: yes
        I2_united_atom -   C_sp3 [LENNARD_JONES] p_0/k_B:  20.97618 [K], p_1: 5.68100 [A], shift/k_B:  -0.63641879 [K], tailcorrection: yes
        I2_united_atom -    H_h2 [ZERO_POTENTIAL]
        I2_united_atom -   H_com [LENNARD_JONES] p_0/k_B: 137.14955 [K], p_1: 3.97100 [A], shift/k_B:  -0.48865971 [K], tailcorrection: yes
        I2_united_atom -   C_co2 [LENNARD_JONES] p_0/k_B: 121.86058 [K], p_1: 3.89100 [A], shift/k_B:  -0.38431556 [K], tailcorrection: yes
        I2_united_atom -   O_co2 [LENNARD_JONES] p_0/k_B: 208.44664 [K], p_1: 4.01600 [A], shift/k_B:  -0.79458949 [K], tailcorrection: yes
        I2_united_atom -    O_o2 [LENNARD_JONES] p_0/k_B: 164.16455 [K], p_1: 4.00100 [A], shift/k_B:  -0.61190725 [K], tailcorrection: yes
        I2_united_atom -   O_com [ZERO_POTENTIAL]
        I2_united_atom -    N_n2 [LENNARD_JONES] p_0/k_B: 140.71247 [K], p_1: 4.14600 [A], shift/k_B:  -0.64924219 [K], tailcorrection: yes
        I2_united_atom -   N_com [ZERO_POTENTIAL]
        I2_united_atom -      Ar [LENNARD_JONES] p_0/k_B: 261.25658 [K], p_1: 4.20100 [A], shift/k_B:  -1.30448881 [K], tailcorrection: yes
        I2_united_atom -      Ow [LENNARD_JONES] p_0/k_B: 222.03187 [K], p_1: 4.03950 [A], shift/k_B:  -0.87649999 [K], tailcorrection: yes
        I2_united_atom -      Hw [LENNARD_JONES] p_0/k_B: 110.34944 [K], p_1: 3.77600 [A], shift/k_B:  -0.29072067 [K], tailcorrection: yes
        I2_united_atom -  C_benz [LENNARD_JONES] p_0/k_B: 129.94229 [K], p_1: 4.29100 [A], shift/k_B:  -0.73669028 [K], tailcorrection: yes
        I2_united_atom -  H_benz [LENNARD_JONES] p_0/k_B: 118.31103 [K], p_1: 3.67100 [A], shift/k_B:  -0.26320248 [K], tailcorrection: yes
        I2_united_atom -   N_dmf [LENNARD_JONES] p_0/k_B: 209.76177 [K], p_1: 4.09100 [A], shift/k_B:  -0.89338823 [K], tailcorrection: yes
        I2_united_atom -  Co_dmf [LENNARD_JONES] p_0/k_B: 165.83124 [K], p_1: 4.34100 [A], shift/k_B:  -1.00772943 [K], tailcorrection: yes
        I2_united_atom -  Cm_dmf [LENNARD_JONES] p_0/k_B: 209.76177 [K], p_1: 4.39100 [A], shift/k_B:  -1.36520782 [K], tailcorrection: yes
        I2_united_atom -   O_dmf [LENNARD_JONES] p_0/k_B: 234.52079 [K], p_1: 3.97100 [A], shift/k_B:  -0.83559047 [K], tailcorrection: yes
        I2_united_atom -   H_dmf [LENNARD_JONES] p_0/k_B:  66.33250 [K], p_1: 3.59100 [A], shift/k_B:  -0.12930257 [K], tailcorrection: yes
        I2_united_atom -      Na [LENNARD_JONES] p_0/k_B:  91.10159 [K], p_1: 3.82100 [A], shift/k_B:  -0.25768013 [K], tailcorrection: yes
        I2_united_atom -      Cl [LENNARD_JONES] p_0/k_B: 250.63021 [K], p_1: 4.25100 [A], shift/k_B:  -1.34337446 [K], tailcorrection: yes
        I2_united_atom -      Kr [LENNARD_JONES] p_0/k_B: 301.42993 [K], p_1: 4.32100 [A], shift/k_B:  -1.78175887 [K], tailcorrection: yes
        I2_united_atom -      Xe [LENNARD_JONES] p_0/k_B: 355.51371 [K], p_1: 4.47600 [A], shift/k_B:  -2.59538738 [K], tailcorrection: yes
        I2_united_atom - O_tip4p_2005 [LENNARD_JONES] p_0/k_B: 226.40671 [K], p_1: 4.07050 [A], shift/k_B:  -0.93567845 [K], tailcorrection: yes
        I2_united_atom - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 2.49100 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        I2_united_atom - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 2.49100 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        I2_united_atom -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B: 124.49498 [K], p_1: 3.85100 [A], shift/k_B:  -0.36903786 [K], tailcorrection: yes
        I2_united_atom -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B: 124.49498 [K], p_1: 3.85100 [A], shift/k_B:  -0.36903786 [K], tailcorrection: yes
        I2_united_atom -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B: 124.49498 [K], p_1: 3.85100 [A], shift/k_B:  -0.36903786 [K], tailcorrection: yes
        I2_united_atom -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B: 124.49498 [K], p_1: 3.85100 [A], shift/k_B:  -0.36903786 [K], tailcorrection: yes
         CH_sp3 -  CH_sp3 [LENNARD_JONES] p_0/k_B:  17.00000 [K], p_1: 4.67000 [A], shift/k_B:  -0.16000180 [K], tailcorrection: yes
         CH_sp3 -   C_sp3 [LENNARD_JONES] p_0/k_B:   3.68782 [K], p_1: 5.52500 [A], shift/k_B:  -0.09478627 [K], tailcorrection: yes
         CH_sp3 -    H_h2 [ZERO_POTENTIAL]
         CH_sp3 -   H_com [LENNARD_JONES] p_0/k_B:  24.11224 [K], p_1: 3.81500 [A], shift/k_B:  -0.06756167 [K], tailcorrection: yes
         CH_sp3 -   C_co2 [LENNARD_JONES] p_0/k_B:  21.42429 [K], p_1: 3.73500 [A], shift/k_B:  -0.05286667 [K], tailcorrection: yes
         CH_sp3 -   O_co2 [LENNARD_JONES] p_0/k_B:  36.64696 [K], p_1: 3.86000 [A], shift/k_B:  -0.11016287 [K], tailcorrection: yes
         CH_sp3 -    O_o2 [LENNARD_JONES] p_0/k_B:  28.86174 [K], p_1: 3.84500 [A], shift/k_B:  -0.08475816 [K], tailcorrection: yes
         CH_sp3 -   O_com [ZERO_POTENTIAL]
         CH_sp3 -    N_n2 [LENNARD_JONES] p_0/k_B:  24.73863 [K], p_1: 3.99000 [A], shift/k_B:  -0.09070153 [K], tailcorrection: yes
         CH_sp3 -   N_com [ZERO_POTENTIAL]
         CH_sp3 -      Ar [LENNARD_JONES] p_0/k_B:  45.93147 [K], p_1: 4.04500 [A], shift/k_B:  -0.18280536 [K], tailcorrection: yes
         CH_sp3 -      Ow [LENNARD_JONES] p_0/k_B:  39.03538 [K], p_1: 3.88350 [A], shift/k_B:  -0.12169132 [K], tailcorrection: yes
         CH_sp3 -      Hw [LENNARD_JONES] p_0/k_B:  19.40052 [K], p_1: 3.62000 [A], shift/k_B:  -0.03968648 [K], tailcorrection: yes
         CH_sp3 -  C_benz [LENNARD_JONES] p_0/k_B:  22.84513 [K], p_1: 4.13500 [A], shift/k_B:  -0.10374162 [K], tailcorrection: yes
         CH_sp3 -  H_benz [LENNARD_JONES] p_0/k_B:  20.80024 [K], p_1: 3.51500 [A], shift/k_B:  -0.03566434 [K], tailcorrection: yes
         CH_sp3 -   N_dmf [LENNARD_JONES] p_0/k_B:  36.87818 [K], p_1: 3.93500 [A], shift/k_B:  -0.12441458 [K], tailcorrection: yes
         CH_sp3 -  Co_dmf [LENNARD_JONES] p_0/k_B:  29.15476 [K], p_1: 4.18500 [A], shift/k_B:  -0.14228253 [K], tailcorrection: yes
         CH_sp3 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  36.87818 [K], p_1: 4.23500 [A], shift/k_B:  -0.19325024 [K], tailcorrection: yes
         CH_sp3 -   O_dmf [LENNARD_JONES] p_0/k_B:  41.23106 [K], p_1: 3.81500 [A], shift/k_B:  -0.11552802 [K], tailcorrection: yes
         CH_sp3 -   H_dmf [LENNARD_JONES] p_0/k_B:  11.66190 [K], p_1: 3.43500 [A], shift/k_B:  -0.01741678 [K], tailcorrection: yes
         CH_sp3 -      Na [LENNARD_JONES] p_0/k_B:  16.01655 [K], p_1: 3.66500 [A], shift/k_B:  -0.03528367 [K], tailcorrection: yes
         CH_sp3 -      Cl [LENNARD_JONES] p_0/k_B:  44.06325 [K], p_1: 4.09500 [A], shift/k_B:  -0.18877055 [K], tailcorrection: yes
         CH_sp3 -      Kr [LENNARD_JONES] p_0/k_B:  52.99434 [K], p_1: 4.16500 [A], shift/k_B:  -0.25130660 [K], tailcorrection: yes
         CH_sp3 -      Xe [LENNARD_JONES] p_0/k_B:  62.50280 [K], p_1: 4.32000 [A], shift/k_B:  -0.36894346 [K], tailcorrection: yes
         CH_sp3 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  39.80452 [K], p_1: 3.91450 [A], shift/k_B:  -0.13014723 [K], tailcorrection: yes
         CH_sp3 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 2.33500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         CH_sp3 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 2.33500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         CH_sp3 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  21.88744 [K], p_1: 3.69500 [A], shift/k_B:  -0.05063262 [K], tailcorrection: yes
         CH_sp3 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  21.88744 [K], p_1: 3.69500 [A], shift/k_B:  -0.05063262 [K], tailcorrection: yes
         CH_sp3 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  21.88744 [K], p_1: 3.69500 [A], shift/k_B:  -0.05063262 [K], tailcorrection: yes
         CH_sp3 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  21.88744 [K], p_1: 3.69500 [A], shift/k_B:  -0.05063262 [K], tailcorrection: yes
          C_sp3 -   C_sp3 [LENNARD_JONES] p_0/k_B:   0.80000 [K], p_1: 6.38000 [A], shift/k_B:  -0.04831734 [K], tailcorrection: yes
          C_sp3 -    H_h2 [ZERO_POTENTIAL]
          C_sp3 -   H_com [LENNARD_JONES] p_0/k_B:   5.23068 [K], p_1: 4.67000 [A], shift/k_B:  -0.04923047 [K], tailcorrection: yes
          C_sp3 -   C_co2 [LENNARD_JONES] p_0/k_B:   4.64758 [K], p_1: 4.59000 [A], shift/k_B:  -0.03944380 [K], tailcorrection: yes
          C_sp3 -   O_co2 [LENNARD_JONES] p_0/k_B:   7.94984 [K], p_1: 4.71500 [A], shift/k_B:  -0.07924330 [K], tailcorrection: yes
          C_sp3 -    O_o2 [LENNARD_JONES] p_0/k_B:   6.26099 [K], p_1: 4.70000 [A], shift/k_B:  -0.06123005 [K], tailcorrection: yes
          C_sp3 -   O_com [ZERO_POTENTIAL]
          C_sp3 -    N_n2 [LENNARD_JONES] p_0/k_B:   5.36656 [K], p_1: 4.84500 [A], shift/k_B:  -0.06294771 [K], tailcorrection: yes
          C_sp3 -   N_com [ZERO_POTENTIAL]
          C_sp3 -      Ar [LENNARD_JONES] p_0/k_B:   9.96393 [K], p_1: 4.90000 [A], shift/k_B:  -0.12503701 [K], tailcorrection: yes
          C_sp3 -      Ow [LENNARD_JONES] p_0/k_B:   8.46796 [K], p_1: 4.73850 [A], shift/k_B:  -0.08695713 [K], tailcorrection: yes
          C_sp3 -      Hw [LENNARD_JONES] p_0/k_B:   4.20856 [K], p_1: 4.47500 [A], shift/k_B:  -0.03068305 [K], tailcorrection: yes
          C_sp3 -  C_benz [LENNARD_JONES] p_0/k_B:   4.95580 [K], p_1: 4.99000 [A], shift/k_B:  -0.06934106 [K], tailcorrection: yes
          C_sp3 -  H_benz [LENNARD_JONES] p_0/k_B:   4.51221 [K], p_1: 4.37000 [A], shift/k_B:  -0.02853576 [K], tailcorrection: yes
          C_sp3 -   N_dmf [LENNARD_JONES] p_0/k_B:   8.00000 [K], p_1: 4.79000 [A], shift/k_B:  -0.08764131 [K], tailcorrection: yes
          C_sp3 -  Co_dmf [LENNARD_JONES] p_0/k_B:   6.32456 [K], p_1: 5.04000 [A], shift/k_B:  -0.09392731 [K], tailcorrection: yes
          C_sp3 -  Cm_dmf [LENNARD_JONES] p_0/k_B:   8.00000 [K], p_1: 5.09000 [A], shift/k_B:  -0.12603066 [K], tailcorrection: yes
          C_sp3 -   O_dmf [LENNARD_JONES] p_0/k_B:   8.94427 [K], p_1: 4.67000 [A], shift/k_B:  -0.08418233 [K], tailcorrection: yes
          C_sp3 -   H_dmf [LENNARD_JONES] p_0/k_B:   2.52982 [K], p_1: 4.29000 [A], shift/k_B:  -0.01432247 [K], tailcorrection: yes
          C_sp3 -      Na [LENNARD_JONES] p_0/k_B:   3.47448 [K], p_1: 4.52000 [A], shift/k_B:  -0.02689537 [K], tailcorrection: yes
          C_sp3 -      Cl [LENNARD_JONES] p_0/k_B:   9.55866 [K], p_1: 4.95000 [A], shift/k_B:  -0.12745984 [K], tailcorrection: yes
          C_sp3 -      Kr [LENNARD_JONES] p_0/k_B:  11.49609 [K], p_1: 5.02000 [A], shift/k_B:  -0.16672061 [K], tailcorrection: yes
          C_sp3 -      Xe [LENNARD_JONES] p_0/k_B:  13.55876 [K], p_1: 5.17500 [A], shift/k_B:  -0.23582047 [K], tailcorrection: yes
          C_sp3 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:   8.63481 [K], p_1: 4.76950 [A], shift/k_B:  -0.09219901 [K], tailcorrection: yes
          C_sp3 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 3.19000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          C_sp3 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 3.19000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          C_sp3 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   4.74805 [K], p_1: 4.55000 [A], shift/k_B:  -0.03823904 [K], tailcorrection: yes
          C_sp3 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   4.74805 [K], p_1: 4.55000 [A], shift/k_B:  -0.03823904 [K], tailcorrection: yes
          C_sp3 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   4.74805 [K], p_1: 4.55000 [A], shift/k_B:  -0.03823904 [K], tailcorrection: yes
          C_sp3 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   4.74805 [K], p_1: 4.55000 [A], shift/k_B:  -0.03823904 [K], tailcorrection: yes
           H_h2 -    H_h2 [ZERO_POTENTIAL]
           H_h2 -   H_com [ZERO_POTENTIAL]
           H_h2 -   C_co2 [ZERO_POTENTIAL]
           H_h2 -   O_co2 [ZERO_POTENTIAL]
           H_h2 -    O_o2 [ZERO_POTENTIAL]
           H_h2 -   O_com [ZERO_POTENTIAL]
           H_h2 -    N_n2 [ZERO_POTENTIAL]
           H_h2 -   N_com [ZERO_POTENTIAL]
           H_h2 -      Ar [ZERO_POTENTIAL]
           H_h2 -      Ow [ZERO_POTENTIAL]
           H_h2 -      Hw [ZERO_POTENTIAL]
           H_h2 -      Lw [ZERO_POTENTIAL]
           H_h2 -  C_benz [ZERO_POTENTIAL]
           H_h2 -  H_benz [ZERO_POTENTIAL]
           H_h2 -   N_dmf [ZERO_POTENTIAL]
           H_h2 -  Co_dmf [ZERO_POTENTIAL]
           H_h2 -  Cm_dmf [ZERO_POTENTIAL]
           H_h2 -   O_dmf [ZERO_POTENTIAL]
           H_h2 -   H_dmf [ZERO_POTENTIAL]
           H_h2 -      Na [ZERO_POTENTIAL]
           H_h2 -      Cl [ZERO_POTENTIAL]
           H_h2 -      Kr [ZERO_POTENTIAL]
           H_h2 -      Xe [ZERO_POTENTIAL]
           H_h2 - O_tip4p_2005 [ZERO_POTENTIAL]
           H_h2 - H_tip4p_2005 [ZERO_POTENTIAL]
           H_h2 - M_tip4p_2005 [ZERO_POTENTIAL]
           H_h2 -  Mof_Zn [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
           H_h2 -   Mof_O [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
           H_h2 -   Mof_C [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
           H_h2 -   Mof_H [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          H_com -   H_com [LENNARD_JONES] p_0/k_B:  34.20000 [K], p_1: 2.96000 [A], shift/k_B:  -0.02091752 [K], tailcorrection: yes
          H_com -   C_co2 [LENNARD_JONES] p_0/k_B:  30.38750 [K], p_1: 2.88000 [A], shift/k_B:  -0.01576862 [K], tailcorrection: yes
          H_com -   O_co2 [LENNARD_JONES] p_0/k_B:  51.97884 [K], p_1: 3.00500 [A], shift/k_B:  -0.03480334 [K], tailcorrection: yes
          H_com -    O_o2 [LENNARD_JONES] p_0/k_B:  40.93654 [K], p_1: 2.99000 [A], shift/k_B:  -0.02659915 [K], tailcorrection: yes
          H_com -   O_com [ZERO_POTENTIAL]
          H_com -    N_n2 [LENNARD_JONES] p_0/k_B:  35.08846 [K], p_1: 3.13500 [A], shift/k_B:  -0.03028977 [K], tailcorrection: yes
          H_com -   N_com [ZERO_POTENTIAL]
          H_com -      Ar [LENNARD_JONES] p_0/k_B:  65.14768 [K], p_1: 3.19000 [A], shift/k_B:  -0.06242221 [K], tailcorrection: yes
          H_com -      Ow [LENNARD_JONES] p_0/k_B:  55.36649 [K], p_1: 3.02850 [A], shift/k_B:  -0.03884512 [K], tailcorrection: yes
          H_com -      Hw [LENNARD_JONES] p_0/k_B:  27.51705 [K], p_1: 2.76500 [A], shift/k_B:  -0.01118224 [K], tailcorrection: yes
          H_com -  C_benz [LENNARD_JONES] p_0/k_B:  32.40278 [K], p_1: 3.28000 [A], shift/k_B:  -0.03668618 [K], tailcorrection: yes
          H_com -  H_benz [LENNARD_JONES] p_0/k_B:  29.50237 [K], p_1: 2.66000 [A], shift/k_B:  -0.00950412 [K], tailcorrection: yes
          H_com -   N_dmf [LENNARD_JONES] p_0/k_B:  52.30679 [K], p_1: 3.08000 [A], shift/k_B:  -0.04060488 [K], tailcorrection: yes
          H_com -  Co_dmf [LENNARD_JONES] p_0/k_B:  41.35215 [K], p_1: 3.33000 [A], shift/k_B:  -0.05126594 [K], tailcorrection: yes
          H_com -  Cm_dmf [LENNARD_JONES] p_0/k_B:  52.30679 [K], p_1: 3.38000 [A], shift/k_B:  -0.07091060 [K], tailcorrection: yes
          H_com -   O_dmf [LENNARD_JONES] p_0/k_B:  58.48077 [K], p_1: 2.96000 [A], shift/k_B:  -0.03576820 [K], tailcorrection: yes
          H_com -   H_dmf [LENNARD_JONES] p_0/k_B:  16.54086 [K], p_1: 2.58000 [A], shift/k_B:  -0.00443657 [K], tailcorrection: yes
          H_com -      Na [LENNARD_JONES] p_0/k_B:  22.71735 [K], p_1: 2.81000 [A], shift/k_B:  -0.01017061 [K], tailcorrection: yes
          H_com -      Cl [LENNARD_JONES] p_0/k_B:  62.49786 [K], p_1: 3.24000 [A], shift/k_B:  -0.06573870 [K], tailcorrection: yes
          H_com -      Kr [LENNARD_JONES] p_0/k_B:  75.16542 [K], p_1: 3.31000 [A], shift/k_B:  -0.08987861 [K], tailcorrection: yes
          H_com -      Xe [LENNARD_JONES] p_0/k_B:  88.65190 [K], p_1: 3.46500 [A], shift/k_B:  -0.13948796 [K], tailcorrection: yes
          H_com - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  56.45742 [K], p_1: 3.05950 [A], shift/k_B:  -0.04210590 [K], tailcorrection: yes
          H_com - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.48000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          H_com - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.48000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          H_com -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.04442 [K], p_1: 2.84000 [A], shift/k_B:  -0.01481296 [K], tailcorrection: yes
          H_com -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.04442 [K], p_1: 2.84000 [A], shift/k_B:  -0.01481296 [K], tailcorrection: yes
          H_com -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.04442 [K], p_1: 2.84000 [A], shift/k_B:  -0.01481296 [K], tailcorrection: yes
          H_com -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.04442 [K], p_1: 2.84000 [A], shift/k_B:  -0.01481296 [K], tailcorrection: yes
          C_co2 -   C_co2 [LENNARD_JONES] p_0/k_B:  27.00000 [K], p_1: 2.80000 [A], shift/k_B:  -0.01183217 [K], tailcorrection: yes
          C_co2 -   O_co2 [LENNARD_JONES] p_0/k_B:  46.18441 [K], p_1: 2.92500 [A], shift/k_B:  -0.02630201 [K], tailcorrection: yes
          C_co2 -    O_o2 [LENNARD_JONES] p_0/k_B:  36.37307 [K], p_1: 2.91000 [A], shift/k_B:  -0.02008529 [K], tailcorrection: yes
          C_co2 -   O_com [ZERO_POTENTIAL]
          C_co2 -    N_n2 [LENNARD_JONES] p_0/k_B:  31.17691 [K], p_1: 3.05500 [A], shift/k_B:  -0.02304731 [K], tailcorrection: yes
          C_co2 -   N_com [ZERO_POTENTIAL]
          C_co2 -      Ar [LENNARD_JONES] p_0/k_B:  57.88523 [K], p_1: 3.11000 [A], shift/k_B:  -0.04762565 [K], tailcorrection: yes
          C_co2 -      Ow [LENNARD_JONES] p_0/k_B:  49.19442 [K], p_1: 2.94850 [A], shift/k_B:  -0.02939395 [K], tailcorrection: yes
          C_co2 -      Hw [LENNARD_JONES] p_0/k_B:  24.44954 [K], p_1: 2.68500 [A], shift/k_B:  -0.00833105 [K], tailcorrection: yes
          C_co2 -  C_benz [LENNARD_JONES] p_0/k_B:  28.79062 [K], p_1: 3.20000 [A], shift/k_B:  -0.02810898 [K], tailcorrection: yes
          C_co2 -  H_benz [LENNARD_JONES] p_0/k_B:  26.21355 [K], p_1: 2.58000 [A], shift/k_B:  -0.00703097 [K], tailcorrection: yes
          C_co2 -   N_dmf [LENNARD_JONES] p_0/k_B:  46.47580 [K], p_1: 3.00000 [A], shift/k_B:  -0.03080935 [K], tailcorrection: yes
          C_co2 -  Co_dmf [LENNARD_JONES] p_0/k_B:  36.74235 [K], p_1: 3.25000 [A], shift/k_B:  -0.03936868 [K], tailcorrection: yes
          C_co2 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  46.47580 [K], p_1: 3.30000 [A], shift/k_B:  -0.05457367 [K], tailcorrection: yes
          C_co2 -   O_dmf [LENNARD_JONES] p_0/k_B:  51.96152 [K], p_1: 2.88000 [A], shift/k_B:  -0.02696377 [K], tailcorrection: yes
          C_co2 -   H_dmf [LENNARD_JONES] p_0/k_B:  14.69694 [K], p_1: 2.50000 [A], shift/k_B:  -0.00326319 [K], tailcorrection: yes
          C_co2 -      Na [LENNARD_JONES] p_0/k_B:  20.18490 [K], p_1: 2.73000 [A], shift/k_B:  -0.00759909 [K], tailcorrection: yes
          C_co2 -      Cl [LENNARD_JONES] p_0/k_B:  55.53080 [K], p_1: 3.16000 [A], shift/k_B:  -0.05027572 [K], tailcorrection: yes
          C_co2 -      Kr [LENNARD_JONES] p_0/k_B:  66.78623 [K], p_1: 3.23000 [A], shift/k_B:  -0.06895886 [K], tailcorrection: yes
          C_co2 -      Xe [LENNARD_JONES] p_0/k_B:  78.76928 [K], p_1: 3.38500 [A], shift/k_B:  -0.10773592 [K], tailcorrection: yes
          C_co2 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  50.16373 [K], p_1: 2.97950 [A], shift/k_B:  -0.03191400 [K], tailcorrection: yes
          C_co2 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.40000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          C_co2 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.40000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          C_co2 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  27.58369 [K], p_1: 2.76000 [A], shift/k_B:  -0.01108826 [K], tailcorrection: yes
          C_co2 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  27.58369 [K], p_1: 2.76000 [A], shift/k_B:  -0.01108826 [K], tailcorrection: yes
          C_co2 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  27.58369 [K], p_1: 2.76000 [A], shift/k_B:  -0.01108826 [K], tailcorrection: yes
          C_co2 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  27.58369 [K], p_1: 2.76000 [A], shift/k_B:  -0.01108826 [K], tailcorrection: yes
          O_co2 -   O_co2 [LENNARD_JONES] p_0/k_B:  79.00000 [K], p_1: 3.05000 [A], shift/k_B:  -0.05782915 [K], tailcorrection: yes
          O_co2 -    O_o2 [LENNARD_JONES] p_0/k_B:  62.21736 [K], p_1: 3.03500 [A], shift/k_B:  -0.04421674 [K], tailcorrection: yes
          O_co2 -   O_com [ZERO_POTENTIAL]
          O_co2 -    N_n2 [LENNARD_JONES] p_0/k_B:  53.32917 [K], p_1: 3.18000 [A], shift/k_B:  -0.05014476 [K], tailcorrection: yes
          O_co2 -   N_com [ZERO_POTENTIAL]
          O_co2 -      Ar [LENNARD_JONES] p_0/k_B:  99.01465 [K], p_1: 3.23500 [A], shift/k_B:  -0.10318870 [K], tailcorrection: yes
          O_co2 -      Ow [LENNARD_JONES] p_0/k_B:  84.14872 [K], p_1: 3.07350 [A], shift/k_B:  -0.06450060 [K], tailcorrection: yes
          O_co2 -      Hw [LENNARD_JONES] p_0/k_B:  41.82176 [K], p_1: 2.81000 [A], shift/k_B:  -0.01872371 [K], tailcorrection: yes
          O_co2 -  C_benz [LENNARD_JONES] p_0/k_B:  49.24733 [K], p_1: 3.32500 [A], shift/k_B:  -0.06050613 [K], tailcorrection: yes
          O_co2 -  H_benz [LENNARD_JONES] p_0/k_B:  44.83916 [K], p_1: 2.70500 [A], shift/k_B:  -0.01597433 [K], tailcorrection: yes
          O_co2 -   N_dmf [LENNARD_JONES] p_0/k_B:  79.49843 [K], p_1: 3.12500 [A], shift/k_B:  -0.06732353 [K], tailcorrection: yes
          O_co2 -  Co_dmf [LENNARD_JONES] p_0/k_B:  62.84903 [K], p_1: 3.37500 [A], shift/k_B:  -0.08444918 [K], tailcorrection: yes
          O_co2 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  79.49843 [K], p_1: 3.42500 [A], shift/k_B:  -0.11667095 [K], tailcorrection: yes
          O_co2 -   O_dmf [LENNARD_JONES] p_0/k_B:  88.88194 [K], p_1: 3.00500 [A], shift/k_B:  -0.05951245 [K], tailcorrection: yes
          O_co2 -   H_dmf [LENNARD_JONES] p_0/k_B:  25.13961 [K], p_1: 2.62500 [A], shift/k_B:  -0.00748001 [K], tailcorrection: yes
          O_co2 -      Na [LENNARD_JONES] p_0/k_B:  34.52695 [K], p_1: 2.85500 [A], shift/k_B:  -0.01700363 [K], tailcorrection: yes
          O_co2 -      Cl [LENNARD_JONES] p_0/k_B:  94.98731 [K], p_1: 3.28500 [A], shift/k_B:  -0.10853103 [K], tailcorrection: yes
          O_co2 -      Kr [LENNARD_JONES] p_0/k_B: 114.24010 [K], p_1: 3.35500 [A], shift/k_B:  -0.14812660 [K], tailcorrection: yes
          O_co2 -      Xe [LENNARD_JONES] p_0/k_B: 134.73752 [K], p_1: 3.51000 [A], shift/k_B:  -0.22905867 [K], tailcorrection: yes
          O_co2 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  85.80676 [K], p_1: 3.10450 [A], shift/k_B:  -0.06985271 [K], tailcorrection: yes
          O_co2 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.52500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          O_co2 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.52500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          O_co2 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.18284 [K], p_1: 2.88500 [A], shift/k_B:  -0.02474014 [K], tailcorrection: yes
          O_co2 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.18284 [K], p_1: 2.88500 [A], shift/k_B:  -0.02474014 [K], tailcorrection: yes
          O_co2 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.18284 [K], p_1: 2.88500 [A], shift/k_B:  -0.02474014 [K], tailcorrection: yes
          O_co2 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.18284 [K], p_1: 2.88500 [A], shift/k_B:  -0.02474014 [K], tailcorrection: yes
           O_o2 -    O_o2 [LENNARD_JONES] p_0/k_B:  49.00000 [K], p_1: 3.02000 [A], shift/k_B:  -0.03380360 [K], tailcorrection: yes
           O_o2 -   O_com [ZERO_POTENTIAL]
           O_o2 -    N_n2 [LENNARD_JONES] p_0/k_B:  42.00000 [K], p_1: 3.16500 [A], shift/k_B:  -0.03838773 [K], tailcorrection: yes
           O_o2 -   N_com [ZERO_POTENTIAL]
           O_o2 -      Ar [LENNARD_JONES] p_0/k_B:  77.98013 [K], p_1: 3.22000 [A], shift/k_B:  -0.07903314 [K], tailcorrection: yes
           O_o2 -      Ow [LENNARD_JONES] p_0/k_B:  66.27229 [K], p_1: 3.05850 [A], shift/k_B:  -0.04932899 [K], tailcorrection: yes
           O_o2 -      Hw [LENNARD_JONES] p_0/k_B:  32.93721 [K], p_1: 2.79500 [A], shift/k_B:  -0.01428009 [K], tailcorrection: yes
           O_o2 -  C_benz [LENNARD_JONES] p_0/k_B:  38.78531 [K], p_1: 3.31000 [A], shift/k_B:  -0.04637730 [K], tailcorrection: yes
           O_o2 -  H_benz [LENNARD_JONES] p_0/k_B:  35.31360 [K], p_1: 2.69000 [A], shift/k_B:  -0.01216798 [K], tailcorrection: yes
           O_o2 -   N_dmf [LENNARD_JONES] p_0/k_B:  62.60990 [K], p_1: 3.11000 [A], shift/k_B:  -0.05151292 [K], tailcorrection: yes
           O_o2 -  Co_dmf [LENNARD_JONES] p_0/k_B:  49.49747 [K], p_1: 3.36000 [A], shift/k_B:  -0.06475552 [K], tailcorrection: yes
           O_o2 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  62.60990 [K], p_1: 3.41000 [A], shift/k_B:  -0.08949818 [K], tailcorrection: yes
           O_o2 -   O_dmf [LENNARD_JONES] p_0/k_B:  70.00000 [K], p_1: 2.99000 [A], shift/k_B:  -0.04548359 [K], tailcorrection: yes
           O_o2 -   H_dmf [LENNARD_JONES] p_0/k_B:  19.79899 [K], p_1: 2.61000 [A], shift/k_B:  -0.00569187 [K], tailcorrection: yes
           O_o2 -      Na [LENNARD_JONES] p_0/k_B:  27.19209 [K], p_1: 2.84000 [A], shift/k_B:  -0.01297481 [K], tailcorrection: yes
           O_o2 -      Cl [LENNARD_JONES] p_0/k_B:  74.80836 [K], p_1: 3.27000 [A], shift/k_B:  -0.08316030 [K], tailcorrection: yes
           O_o2 -      Kr [LENNARD_JONES] p_0/k_B:  89.97111 [K], p_1: 3.34000 [A], shift/k_B:  -0.11356511 [K], tailcorrection: yes
           O_o2 -      Xe [LENNARD_JONES] p_0/k_B: 106.11409 [K], p_1: 3.49500 [A], shift/k_B:  -0.17582325 [K], tailcorrection: yes
           O_o2 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  67.57810 [K], p_1: 3.08950 [A], shift/k_B:  -0.05343791 [K], tailcorrection: yes
           O_o2 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.51000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
           O_o2 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.51000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
           O_o2 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.15939 [K], p_1: 2.87000 [A], shift/k_B:  -0.01888447 [K], tailcorrection: yes
           O_o2 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.15939 [K], p_1: 2.87000 [A], shift/k_B:  -0.01888447 [K], tailcorrection: yes
           O_o2 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.15939 [K], p_1: 2.87000 [A], shift/k_B:  -0.01888447 [K], tailcorrection: yes
           O_o2 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.15939 [K], p_1: 2.87000 [A], shift/k_B:  -0.01888447 [K], tailcorrection: yes
          O_com -   O_com [ZERO_POTENTIAL]
          O_com -    N_n2 [ZERO_POTENTIAL]
          O_com -   N_com [ZERO_POTENTIAL]
          O_com -      Ar [ZERO_POTENTIAL]
          O_com -      Ow [ZERO_POTENTIAL]
          O_com -      Hw [ZERO_POTENTIAL]
          O_com -      Lw [ZERO_POTENTIAL]
          O_com -  C_benz [ZERO_POTENTIAL]
          O_com -  H_benz [ZERO_POTENTIAL]
          O_com -   N_dmf [ZERO_POTENTIAL]
          O_com -  Co_dmf [ZERO_POTENTIAL]
          O_com -  Cm_dmf [ZERO_POTENTIAL]
          O_com -   O_dmf [ZERO_POTENTIAL]
          O_com -   H_dmf [ZERO_POTENTIAL]
          O_com -      Na [ZERO_POTENTIAL]
          O_com -      Cl [ZERO_POTENTIAL]
          O_com -      Kr [ZERO_POTENTIAL]
          O_com -      Xe [ZERO_POTENTIAL]
          O_com - O_tip4p_2005 [ZERO_POTENTIAL]
          O_com - H_tip4p_2005 [ZERO_POTENTIAL]
          O_com - M_tip4p_2005 [ZERO_POTENTIAL]
          O_com -  Mof_Zn [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          O_com -   Mof_O [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          O_com -   Mof_C [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          O_com -   Mof_H [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
           N_n2 -    N_n2 [LENNARD_JONES] p_0/k_B:  36.00000 [K], p_1: 3.31000 [A], shift/k_B:  -0.04304679 [K], tailcorrection: yes
           N_n2 -   N_com [ZERO_POTENTIAL]
           N_n2 -      Ar [LENNARD_JONES] p_0/k_B:  66.84011 [K], p_1: 3.36500 [A], shift/k_B:  -0.08822757 [K], tailcorrection: yes
           N_n2 -      Ow [LENNARD_JONES] p_0/k_B:  56.80482 [K], p_1: 3.20350 [A], shift/k_B:  -0.05582478 [K], tailcorrection: yes
           N_n2 -      Hw [LENNARD_JONES] p_0/k_B:  28.23190 [K], p_1: 2.94000 [A], shift/k_B:  -0.01657908 [K], tailcorrection: yes
           N_n2 -  C_benz [LENNARD_JONES] p_0/k_B:  33.24455 [K], p_1: 3.45500 [A], shift/k_B:  -0.05140921 [K], tailcorrection: yes
           N_n2 -  H_benz [LENNARD_JONES] p_0/k_B:  30.26880 [K], p_1: 2.83500 [A], shift/k_B:  -0.01429099 [K], tailcorrection: yes
           N_n2 -   N_dmf [LENNARD_JONES] p_0/k_B:  53.66563 [K], p_1: 3.25500 [A], shift/k_B:  -0.05803432 [K], tailcorrection: yes
           N_n2 -  Co_dmf [LENNARD_JONES] p_0/k_B:  42.42641 [K], p_1: 3.50500 [A], shift/k_B:  -0.07151241 [K], tailcorrection: yes
           N_n2 -  Cm_dmf [LENNARD_JONES] p_0/k_B:  53.66563 [K], p_1: 3.55500 [A], shift/k_B:  -0.09847697 [K], tailcorrection: yes
           N_n2 -   O_dmf [LENNARD_JONES] p_0/k_B:  60.00000 [K], p_1: 3.13500 [A], shift/k_B:  -0.05179442 [K], tailcorrection: yes
           N_n2 -   H_dmf [LENNARD_JONES] p_0/k_B:  16.97056 [K], p_1: 2.75500 [A], shift/k_B:  -0.00674812 [K], tailcorrection: yes
           N_n2 -      Na [LENNARD_JONES] p_0/k_B:  23.30751 [K], p_1: 2.98500 [A], shift/k_B:  -0.01499313 [K], tailcorrection: yes
           N_n2 -      Cl [LENNARD_JONES] p_0/k_B:  64.12145 [K], p_1: 3.41500 [A], shift/k_B:  -0.09246792 [K], tailcorrection: yes
           N_n2 -      Kr [LENNARD_JONES] p_0/k_B:  77.11809 [K], p_1: 3.48500 [A], shift/k_B:  -0.12560190 [K], tailcorrection: yes
           N_n2 -      Xe [LENNARD_JONES] p_0/k_B:  90.95493 [K], p_1: 3.64000 [A], shift/k_B:  -0.19231138 [K], tailcorrection: yes
           N_n2 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  57.92409 [K], p_1: 3.23450 [A], shift/k_B:  -0.06030999 [K], tailcorrection: yes
           N_n2 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.65500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
           N_n2 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.65500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
           N_n2 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.85090 [K], p_1: 3.01500 [A], shift/k_B:  -0.02175563 [K], tailcorrection: yes
           N_n2 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.85090 [K], p_1: 3.01500 [A], shift/k_B:  -0.02175563 [K], tailcorrection: yes
           N_n2 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.85090 [K], p_1: 3.01500 [A], shift/k_B:  -0.02175563 [K], tailcorrection: yes
           N_n2 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  31.85090 [K], p_1: 3.01500 [A], shift/k_B:  -0.02175563 [K], tailcorrection: yes
          N_com -   N_com [ZERO_POTENTIAL]
          N_com -      Ar [ZERO_POTENTIAL]
          N_com -      Ow [ZERO_POTENTIAL]
          N_com -      Hw [ZERO_POTENTIAL]
          N_com -      Lw [ZERO_POTENTIAL]
          N_com -  C_benz [ZERO_POTENTIAL]
          N_com -  H_benz [ZERO_POTENTIAL]
          N_com -   N_dmf [ZERO_POTENTIAL]
          N_com -  Co_dmf [ZERO_POTENTIAL]
          N_com -  Cm_dmf [ZERO_POTENTIAL]
          N_com -   O_dmf [ZERO_POTENTIAL]
          N_com -   H_dmf [ZERO_POTENTIAL]
          N_com -      Na [ZERO_POTENTIAL]
          N_com -      Cl [ZERO_POTENTIAL]
          N_com -      Kr [ZERO_POTENTIAL]
          N_com -      Xe [ZERO_POTENTIAL]
          N_com - O_tip4p_2005 [ZERO_POTENTIAL]
          N_com - H_tip4p_2005 [ZERO_POTENTIAL]
          N_com - M_tip4p_2005 [ZERO_POTENTIAL]
          N_com -  Mof_Zn [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          N_com -   Mof_O [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          N_com -   Mof_C [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
          N_com -   Mof_H [ZERO_POTENTIAL_CONTINUOUS_FRACTIONAL]
             Ar -      Ar [LENNARD_JONES] p_0/k_B: 124.10000 [K], p_1: 3.42000 [A], shift/k_B:  -0.18053880 [K], tailcorrection: yes
             Ar -      Ow [LENNARD_JONES] p_0/k_B: 105.46779 [K], p_1: 3.25850 [A], shift/k_B:  -0.11479109 [K], tailcorrection: yes
             Ar -      Hw [LENNARD_JONES] p_0/k_B:  52.41731 [K], p_1: 2.99500 [A], shift/k_B:  -0.03440207 [K], tailcorrection: yes
             Ar -  C_benz [LENNARD_JONES] p_0/k_B:  61.72414 [K], p_1: 3.51000 [A], shift/k_B:  -0.10493328 [K], tailcorrection: yes
             Ar -  H_benz [LENNARD_JONES] p_0/k_B:  56.19915 [K], p_1: 2.89000 [A], shift/k_B:  -0.02977553 [K], tailcorrection: yes
             Ar -   N_dmf [LENNARD_JONES] p_0/k_B:  99.63935 [K], p_1: 3.31000 [A], shift/k_B:  -0.11914317 [K], tailcorrection: yes
             Ar -  Co_dmf [LENNARD_JONES] p_0/k_B:  78.77182 [K], p_1: 3.56000 [A], shift/k_B:  -0.14577062 [K], tailcorrection: yes
             Ar -  Cm_dmf [LENNARD_JONES] p_0/k_B:  99.63935 [K], p_1: 3.61000 [A], shift/k_B:  -0.20047289 [K], tailcorrection: yes
             Ar -   O_dmf [LENNARD_JONES] p_0/k_B: 111.40018 [K], p_1: 3.19000 [A], shift/k_B:  -0.10673973 [K], tailcorrection: yes
             Ar -   H_dmf [LENNARD_JONES] p_0/k_B:  31.50873 [K], p_1: 2.81000 [A], shift/k_B:  -0.01410654 [K], tailcorrection: yes
             Ar -      Na [LENNARD_JONES] p_0/k_B:  43.27435 [K], p_1: 3.04000 [A], shift/k_B:  -0.03105948 [K], tailcorrection: yes
             Ar -      Cl [LENNARD_JONES] p_0/k_B: 119.05235 [K], p_1: 3.47000 [A], shift/k_B:  -0.18894811 [K], tailcorrection: yes
             Ar -      Kr [LENNARD_JONES] p_0/k_B: 143.18282 [K], p_1: 3.54000 [A], shift/k_B:  -0.25616293 [K], tailcorrection: yes
             Ar -      Xe [LENNARD_JONES] p_0/k_B: 168.87327 [K], p_1: 3.69500 [A], shift/k_B:  -0.39065766 [K], tailcorrection: yes
             Ar - O_tip4p_2005 [LENNARD_JONES] p_0/k_B: 107.54590 [K], p_1: 3.28950 [A], shift/k_B:  -0.12389342 [K], tailcorrection: yes
             Ar - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.71000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Ar - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.71000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Ar -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  59.13660 [K], p_1: 3.07000 [A], shift/k_B:  -0.04501986 [K], tailcorrection: yes
             Ar -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  59.13660 [K], p_1: 3.07000 [A], shift/k_B:  -0.04501986 [K], tailcorrection: yes
             Ar -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  59.13660 [K], p_1: 3.07000 [A], shift/k_B:  -0.04501986 [K], tailcorrection: yes
             Ar -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  59.13660 [K], p_1: 3.07000 [A], shift/k_B:  -0.04501986 [K], tailcorrection: yes
             Ow -      Ow [LENNARD_JONES] p_0/k_B:  89.63300 [K], p_1: 3.09700 [A], shift/k_B:  -0.07191644 [K], tailcorrection: yes
             Ow -      Hw [LENNARD_JONES] p_0/k_B:  44.54744 [K], p_1: 2.83350 [A], shift/k_B:  -0.02096579 [K], tailcorrection: yes
             Ow -  C_benz [LENNARD_JONES] p_0/k_B:  52.45696 [K], p_1: 3.34850 [A], shift/k_B:  -0.06723044 [K], tailcorrection: yes
             Ow -  H_benz [LENNARD_JONES] p_0/k_B:  47.76149 [K], p_1: 2.72850 [A], shift/k_B:  -0.01792178 [K], tailcorrection: yes
             Ow -   N_dmf [LENNARD_JONES] p_0/k_B:  84.67963 [K], p_1: 3.14850 [A], shift/k_B:  -0.07500757 [K], tailcorrection: yes
             Ow -  Co_dmf [LENNARD_JONES] p_0/k_B:  66.94513 [K], p_1: 3.39850 [A], shift/k_B:  -0.09377577 [K], tailcorrection: yes
             Ow -  Cm_dmf [LENNARD_JONES] p_0/k_B:  84.67963 [K], p_1: 3.44850 [A], shift/k_B:  -0.12947753 [K], tailcorrection: yes
             Ow -   O_dmf [LENNARD_JONES] p_0/k_B:  94.67471 [K], p_1: 3.02850 [A], shift/k_B:  -0.06642375 [K], tailcorrection: yes
             Ow -   H_dmf [LENNARD_JONES] p_0/k_B:  26.77805 [K], p_1: 2.64850 [A], shift/k_B:  -0.00840514 [K], tailcorrection: yes
             Ow -      Na [LENNARD_JONES] p_0/k_B:  36.77719 [K], p_1: 2.87850 [A], shift/k_B:  -0.01902480 [K], tailcorrection: yes
             Ow -      Cl [LENNARD_JONES] p_0/k_B: 101.17799 [K], p_1: 3.30850 [A], shift/k_B:  -0.12065450 [K], tailcorrection: yes
             Ow -      Kr [LENNARD_JONES] p_0/k_B: 121.68554 [K], p_1: 3.37850 [A], shift/k_B:  -0.16452649 [K], tailcorrection: yes
             Ow -      Xe [LENNARD_JONES] p_0/k_B: 143.51886 [K], p_1: 3.53350 [A], shift/k_B:  -0.25394957 [K], tailcorrection: yes
             Ow - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  91.39910 [K], p_1: 3.12800 [A], shift/k_B:  -0.07784847 [K], tailcorrection: yes
             Ow - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.54850 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Ow - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.54850 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Ow -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  50.25791 [K], p_1: 2.90850 [A], shift/k_B:  -0.02766682 [K], tailcorrection: yes
             Ow -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  50.25791 [K], p_1: 2.90850 [A], shift/k_B:  -0.02766682 [K], tailcorrection: yes
             Ow -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  50.25791 [K], p_1: 2.90850 [A], shift/k_B:  -0.02766682 [K], tailcorrection: yes
             Ow -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  50.25791 [K], p_1: 2.90850 [A], shift/k_B:  -0.02766682 [K], tailcorrection: yes
             Hw -      Hw [LENNARD_JONES] p_0/k_B:  22.14000 [K], p_1: 2.57000 [A], shift/k_B:  -0.00580160 [K], tailcorrection: yes
             Hw -  C_benz [LENNARD_JONES] p_0/k_B:  26.07102 [K], p_1: 3.08500 [A], shift/k_B:  -0.02043638 [K], tailcorrection: yes
             Hw -  H_benz [LENNARD_JONES] p_0/k_B:  23.73738 [K], p_1: 2.46500 [A], shift/k_B:  -0.00484298 [K], tailcorrection: yes
             Hw -   N_dmf [LENNARD_JONES] p_0/k_B:  42.08563 [K], p_1: 2.88500 [A], shift/k_B:  -0.02206744 [K], tailcorrection: yes
             Hw -  Co_dmf [LENNARD_JONES] p_0/k_B:  33.27161 [K], p_1: 3.13500 [A], shift/k_B:  -0.02872139 [K], tailcorrection: yes
             Hw -  Cm_dmf [LENNARD_JONES] p_0/k_B:  42.08563 [K], p_1: 3.18500 [A], shift/k_B:  -0.03994731 [K], tailcorrection: yes
             Hw -   O_dmf [LENNARD_JONES] p_0/k_B:  47.05316 [K], p_1: 2.76500 [A], shift/k_B:  -0.01912122 [K], tailcorrection: yes
             Hw -   H_dmf [LENNARD_JONES] p_0/k_B:  13.30864 [K], p_1: 2.38500 [A], shift/k_B:  -0.00222765 [K], tailcorrection: yes
             Hw -      Na [LENNARD_JONES] p_0/k_B:  18.27820 [K], p_1: 2.61500 [A], shift/k_B:  -0.00531535 [K], tailcorrection: yes
             Hw -      Cl [LENNARD_JONES] p_0/k_B:  50.28528 [K], p_1: 3.04500 [A], shift/k_B:  -0.03644904 [K], tailcorrection: yes
             Hw -      Kr [LENNARD_JONES] p_0/k_B:  60.47750 [K], p_1: 3.11500 [A], shift/k_B:  -0.05024028 [K], tailcorrection: yes
             Hw -      Xe [LENNARD_JONES] p_0/k_B:  71.32862 [K], p_1: 3.27000 [A], shift/k_B:  -0.07929207 [K], tailcorrection: yes
             Hw - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  45.42519 [K], p_1: 2.86450 [A], shift/k_B:  -0.02282104 [K], tailcorrection: yes
             Hw - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.28500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Hw - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.28500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Hw -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  24.97809 [K], p_1: 2.64500 [A], shift/k_B:  -0.00777821 [K], tailcorrection: yes
             Hw -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  24.97809 [K], p_1: 2.64500 [A], shift/k_B:  -0.00777821 [K], tailcorrection: yes
             Hw -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  24.97809 [K], p_1: 2.64500 [A], shift/k_B:  -0.00777821 [K], tailcorrection: yes
             Hw -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  24.97809 [K], p_1: 2.64500 [A], shift/k_B:  -0.00777821 [K], tailcorrection: yes
         C_benz -  C_benz [LENNARD_JONES] p_0/k_B:  30.70000 [K], p_1: 3.60000 [A], shift/k_B:  -0.06074892 [K], tailcorrection: yes
         C_benz -  H_benz [LENNARD_JONES] p_0/k_B:  27.95201 [K], p_1: 2.98000 [A], shift/k_B:  -0.01780089 [K], tailcorrection: yes
         C_benz -   N_dmf [LENNARD_JONES] p_0/k_B:  49.55805 [K], p_1: 3.40000 [A], shift/k_B:  -0.06960417 [K], tailcorrection: yes
         C_benz -  Co_dmf [LENNARD_JONES] p_0/k_B:  39.17908 [K], p_1: 3.65000 [A], shift/k_B:  -0.08421278 [K], tailcorrection: yes
         C_benz -  Cm_dmf [LENNARD_JONES] p_0/k_B:  49.55805 [K], p_1: 3.70000 [A], shift/k_B:  -0.11557696 [K], tailcorrection: yes
         C_benz -   O_dmf [LENNARD_JONES] p_0/k_B:  55.40758 [K], p_1: 3.28000 [A], shift/k_B:  -0.06273204 [K], tailcorrection: yes
         C_benz -   H_dmf [LENNARD_JONES] p_0/k_B:  15.67163 [K], p_1: 2.90000 [A], shift/k_B:  -0.00847703 [K], tailcorrection: yes
         C_benz -      Na [LENNARD_JONES] p_0/k_B:  21.52355 [K], p_1: 3.13000 [A], shift/k_B:  -0.01840294 [K], tailcorrection: yes
         C_benz -      Cl [LENNARD_JONES] p_0/k_B:  59.21357 [K], p_1: 3.56000 [A], shift/k_B:  -0.10957724 [K], tailcorrection: yes
         C_benz -      Kr [LENNARD_JONES] p_0/k_B:  71.21545 [K], p_1: 3.63000 [A], shift/k_B:  -0.14811128 [K], tailcorrection: yes
         C_benz -      Xe [LENNARD_JONES] p_0/k_B:  83.99321 [K], p_1: 3.78500 [A], shift/k_B:  -0.22446531 [K], tailcorrection: yes
         C_benz - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  53.49056 [K], p_1: 3.37950 [A], shift/k_B:  -0.07245108 [K], tailcorrection: yes
         C_benz - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.80000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         C_benz - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.80000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         C_benz -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  29.41302 [K], p_1: 3.16000 [A], shift/k_B:  -0.02662956 [K], tailcorrection: yes
         C_benz -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  29.41302 [K], p_1: 3.16000 [A], shift/k_B:  -0.02662956 [K], tailcorrection: yes
         C_benz -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  29.41302 [K], p_1: 3.16000 [A], shift/k_B:  -0.02662956 [K], tailcorrection: yes
         C_benz -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  29.41302 [K], p_1: 3.16000 [A], shift/k_B:  -0.02662956 [K], tailcorrection: yes
         H_benz -  H_benz [LENNARD_JONES] p_0/k_B:  25.45000 [K], p_1: 2.36000 [A], shift/k_B:  -0.00399892 [K], tailcorrection: yes
         H_benz -   N_dmf [LENNARD_JONES] p_0/k_B:  45.12206 [K], p_1: 2.78000 [A], shift/k_B:  -0.01894141 [K], tailcorrection: yes
         H_benz -  Co_dmf [LENNARD_JONES] p_0/k_B:  35.67212 [K], p_1: 3.03000 [A], shift/k_B:  -0.02510201 [K], tailcorrection: yes
         H_benz -  Cm_dmf [LENNARD_JONES] p_0/k_B:  45.12206 [K], p_1: 3.08000 [A], shift/k_B:  -0.03502749 [K], tailcorrection: yes
         H_benz -   O_dmf [LENNARD_JONES] p_0/k_B:  50.44799 [K], p_1: 2.66000 [A], shift/k_B:  -0.01625170 [K], tailcorrection: yes
         H_benz -   H_dmf [LENNARD_JONES] p_0/k_B:  14.26885 [K], p_1: 2.28000 [A], shift/k_B:  -0.00182299 [K], tailcorrection: yes
         H_benz -      Na [LENNARD_JONES] p_0/k_B:  19.59695 [K], p_1: 2.51000 [A], shift/k_B:  -0.00445663 [K], tailcorrection: yes
         H_benz -      Cl [LENNARD_JONES] p_0/k_B:  53.91331 [K], p_1: 2.94000 [A], shift/k_B:  -0.03166040 [K], tailcorrection: yes
         H_benz -      Kr [LENNARD_JONES] p_0/k_B:  64.84088 [K], p_1: 3.01000 [A], shift/k_B:  -0.04385051 [K], tailcorrection: yes
         H_benz -      Xe [LENNARD_JONES] p_0/k_B:  76.47490 [K], p_1: 3.16500 [A], shift/k_B:  -0.06989757 [K], tailcorrection: yes
         H_benz - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  48.70257 [K], p_1: 2.75950 [A], shift/k_B:  -0.01955649 [K], tailcorrection: yes
         H_benz - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.18000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         H_benz - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.18000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         H_benz -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  26.78024 [K], p_1: 2.54000 [A], shift/k_B:  -0.00654019 [K], tailcorrection: yes
         H_benz -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  26.78024 [K], p_1: 2.54000 [A], shift/k_B:  -0.00654019 [K], tailcorrection: yes
         H_benz -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  26.78024 [K], p_1: 2.54000 [A], shift/k_B:  -0.00654019 [K], tailcorrection: yes
         H_benz -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  26.78024 [K], p_1: 2.54000 [A], shift/k_B:  -0.00654019 [K], tailcorrection: yes
          N_dmf -   N_dmf [LENNARD_JONES] p_0/k_B:  80.00000 [K], p_1: 3.20000 [A], shift/k_B:  -0.07810593 [K], tailcorrection: yes
          N_dmf -  Co_dmf [LENNARD_JONES] p_0/k_B:  63.24555 [K], p_1: 3.45000 [A], shift/k_B:  -0.09695678 [K], tailcorrection: yes
          N_dmf -  Cm_dmf [LENNARD_JONES] p_0/k_B:  80.00000 [K], p_1: 3.50000 [A], shift/k_B:  -0.13369552 [K], tailcorrection: yes
          N_dmf -   O_dmf [LENNARD_JONES] p_0/k_B:  89.44272 [K], p_1: 3.08000 [A], shift/k_B:  -0.06943288 [K], tailcorrection: yes
          N_dmf -   H_dmf [LENNARD_JONES] p_0/k_B:  25.29822 [K], p_1: 2.70000 [A], shift/k_B:  -0.00891322 [K], tailcorrection: yes
          N_dmf -      Na [LENNARD_JONES] p_0/k_B:  34.74478 [K], p_1: 2.93000 [A], shift/k_B:  -0.01999093 [K], tailcorrection: yes
          N_dmf -      Cl [LENNARD_JONES] p_0/k_B:  95.58661 [K], p_1: 3.36000 [A], shift/k_B:  -0.12505204 [K], tailcorrection: yes
          N_dmf -      Kr [LENNARD_JONES] p_0/k_B: 114.96086 [K], p_1: 3.43000 [A], shift/k_B:  -0.17019785 [K], tailcorrection: yes
          N_dmf -      Xe [LENNARD_JONES] p_0/k_B: 135.58761 [K], p_1: 3.58500 [A], shift/k_B:  -0.26166489 [K], tailcorrection: yes
          N_dmf - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  86.34813 [K], p_1: 3.17950 [A], shift/k_B:  -0.08111554 [K], tailcorrection: yes
          N_dmf - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.60000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          N_dmf - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.60000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          N_dmf -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 2.96000 [A], shift/k_B:  -0.02904019 [K], tailcorrection: yes
          N_dmf -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 2.96000 [A], shift/k_B:  -0.02904019 [K], tailcorrection: yes
          N_dmf -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 2.96000 [A], shift/k_B:  -0.02904019 [K], tailcorrection: yes
          N_dmf -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 2.96000 [A], shift/k_B:  -0.02904019 [K], tailcorrection: yes
         Co_dmf -  Co_dmf [LENNARD_JONES] p_0/k_B:  50.00000 [K], p_1: 3.70000 [A], shift/k_B:  -0.11660766 [K], tailcorrection: yes
         Co_dmf -  Cm_dmf [LENNARD_JONES] p_0/k_B:  63.24555 [K], p_1: 3.75000 [A], shift/k_B:  -0.15986120 [K], tailcorrection: yes
         Co_dmf -   O_dmf [LENNARD_JONES] p_0/k_B:  70.71068 [K], p_1: 3.33000 [A], shift/k_B:  -0.08766292 [K], tailcorrection: yes
         Co_dmf -   H_dmf [LENNARD_JONES] p_0/k_B:  20.00000 [K], p_1: 2.95000 [A], shift/k_B:  -0.01198663 [K], tailcorrection: yes
         Co_dmf -      Na [LENNARD_JONES] p_0/k_B:  27.46816 [K], p_1: 3.18000 [A], shift/k_B:  -0.02582798 [K], tailcorrection: yes
         Co_dmf -      Cl [LENNARD_JONES] p_0/k_B:  75.56785 [K], p_1: 3.61000 [A], shift/k_B:  -0.15204139 [K], tailcorrection: yes
         Co_dmf -      Kr [LENNARD_JONES] p_0/k_B:  90.88454 [K], p_1: 3.68000 [A], shift/k_B:  -0.20517847 [K], tailcorrection: yes
         Co_dmf -      Xe [LENNARD_JONES] p_0/k_B: 107.19142 [K], p_1: 3.83500 [A], shift/k_B:  -0.30991183 [K], tailcorrection: yes
         Co_dmf - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  68.26419 [K], p_1: 3.42950 [A], shift/k_B:  -0.10097580 [K], tailcorrection: yes
         Co_dmf - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.85000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         Co_dmf - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.85000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         Co_dmf -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.53665 [K], p_1: 3.21000 [A], shift/k_B:  -0.03734030 [K], tailcorrection: yes
         Co_dmf -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.53665 [K], p_1: 3.21000 [A], shift/k_B:  -0.03734030 [K], tailcorrection: yes
         Co_dmf -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.53665 [K], p_1: 3.21000 [A], shift/k_B:  -0.03734030 [K], tailcorrection: yes
         Co_dmf -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  37.53665 [K], p_1: 3.21000 [A], shift/k_B:  -0.03734030 [K], tailcorrection: yes
         Cm_dmf -  Cm_dmf [LENNARD_JONES] p_0/k_B:  80.00000 [K], p_1: 3.80000 [A], shift/k_B:  -0.21892448 [K], tailcorrection: yes
         Cm_dmf -   O_dmf [LENNARD_JONES] p_0/k_B:  89.44272 [K], p_1: 3.38000 [A], shift/k_B:  -0.12125457 [K], tailcorrection: yes
         Cm_dmf -   H_dmf [LENNARD_JONES] p_0/k_B:  25.29822 [K], p_1: 3.00000 [A], shift/k_B:  -0.01677049 [K], tailcorrection: yes
         Cm_dmf -      Na [LENNARD_JONES] p_0/k_B:  34.74478 [K], p_1: 3.23000 [A], shift/k_B:  -0.03587507 [K], tailcorrection: yes
         Cm_dmf -      Cl [LENNARD_JONES] p_0/k_B:  95.58661 [K], p_1: 3.66000 [A], shift/k_B:  -0.20885569 [K], tailcorrection: yes
         Cm_dmf -      Kr [LENNARD_JONES] p_0/k_B: 114.96086 [K], p_1: 3.73000 [A], shift/k_B:  -0.28140847 [K], tailcorrection: yes
         Cm_dmf -      Xe [LENNARD_JONES] p_0/k_B: 135.58761 [K], p_1: 3.88500 [A], shift/k_B:  -0.42366897 [K], tailcorrection: yes
         Cm_dmf - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  86.34813 [K], p_1: 3.47950 [A], shift/k_B:  -0.13930891 [K], tailcorrection: yes
         Cm_dmf - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.90000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         Cm_dmf - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.90000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         Cm_dmf -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 3.26000 [A], shift/k_B:  -0.05182063 [K], tailcorrection: yes
         Cm_dmf -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 3.26000 [A], shift/k_B:  -0.05182063 [K], tailcorrection: yes
         Cm_dmf -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 3.26000 [A], shift/k_B:  -0.05182063 [K], tailcorrection: yes
         Cm_dmf -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  47.48052 [K], p_1: 3.26000 [A], shift/k_B:  -0.05182063 [K], tailcorrection: yes
          O_dmf -   O_dmf [LENNARD_JONES] p_0/k_B: 100.00000 [K], p_1: 2.96000 [A], shift/k_B:  -0.06116233 [K], tailcorrection: yes
          O_dmf -   H_dmf [LENNARD_JONES] p_0/k_B:  28.28427 [K], p_1: 2.58000 [A], shift/k_B:  -0.00758637 [K], tailcorrection: yes
          O_dmf -      Na [LENNARD_JONES] p_0/k_B:  38.84585 [K], p_1: 2.81000 [A], shift/k_B:  -0.01739138 [K], tailcorrection: yes
          O_dmf -      Cl [LENNARD_JONES] p_0/k_B: 106.86908 [K], p_1: 3.24000 [A], shift/k_B:  -0.11241080 [K], tailcorrection: yes
          O_dmf -      Kr [LENNARD_JONES] p_0/k_B: 128.53015 [K], p_1: 3.31000 [A], shift/k_B:  -0.15368917 [K], tailcorrection: yes
          O_dmf -      Xe [LENNARD_JONES] p_0/k_B: 151.59156 [K], p_1: 3.46500 [A], shift/k_B:  -0.23851939 [K], tailcorrection: yes
          O_dmf - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  96.54015 [K], p_1: 3.05950 [A], shift/k_B:  -0.07199957 [K], tailcorrection: yes
          O_dmf - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.48000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          O_dmf - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.48000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          O_dmf -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  53.08484 [K], p_1: 2.84000 [A], shift/k_B:  -0.02532963 [K], tailcorrection: yes
          O_dmf -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  53.08484 [K], p_1: 2.84000 [A], shift/k_B:  -0.02532963 [K], tailcorrection: yes
          O_dmf -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  53.08484 [K], p_1: 2.84000 [A], shift/k_B:  -0.02532963 [K], tailcorrection: yes
          O_dmf -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  53.08484 [K], p_1: 2.84000 [A], shift/k_B:  -0.02532963 [K], tailcorrection: yes
          H_dmf -   H_dmf [LENNARD_JONES] p_0/k_B:   8.00000 [K], p_1: 2.20000 [A], shift/k_B:  -0.00082493 [K], tailcorrection: yes
          H_dmf -      Na [LENNARD_JONES] p_0/k_B:  10.98727 [K], p_1: 2.43000 [A], shift/k_B:  -0.00205734 [K], tailcorrection: yes
          H_dmf -      Cl [LENNARD_JONES] p_0/k_B:  30.22714 [K], p_1: 2.86000 [A], shift/k_B:  -0.01504317 [K], tailcorrection: yes
          H_dmf -      Kr [LENNARD_JONES] p_0/k_B:  36.35382 [K], p_1: 2.93000 [A], shift/k_B:  -0.02091671 [K], tailcorrection: yes
          H_dmf -      Xe [LENNARD_JONES] p_0/k_B:  42.87657 [K], p_1: 3.08500 [A], shift/k_B:  -0.03360981 [K], tailcorrection: yes
          H_dmf - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  27.30568 [K], p_1: 2.67950 [A], shift/k_B:  -0.00919050 [K], tailcorrection: yes
          H_dmf - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.10000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          H_dmf - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.10000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
          H_dmf -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  15.01466 [K], p_1: 2.46000 [A], shift/k_B:  -0.00302625 [K], tailcorrection: yes
          H_dmf -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  15.01466 [K], p_1: 2.46000 [A], shift/k_B:  -0.00302625 [K], tailcorrection: yes
          H_dmf -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  15.01466 [K], p_1: 2.46000 [A], shift/k_B:  -0.00302625 [K], tailcorrection: yes
          H_dmf -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  15.01466 [K], p_1: 2.46000 [A], shift/k_B:  -0.00302625 [K], tailcorrection: yes
             Na -      Na [LENNARD_JONES] p_0/k_B:  15.09000 [K], p_1: 2.66000 [A], shift/k_B:  -0.00486121 [K], tailcorrection: yes
             Na -      Cl [LENNARD_JONES] p_0/k_B:  41.51420 [K], p_1: 3.09000 [A], shift/k_B:  -0.03285956 [K], tailcorrection: yes
             Na -      Kr [LENNARD_JONES] p_0/k_B:  49.92863 [K], p_1: 3.16000 [A], shift/k_B:  -0.04520370 [K], tailcorrection: yes
             Na -      Xe [LENNARD_JONES] p_0/k_B:  58.88703 [K], p_1: 3.31500 [A], shift/k_B:  -0.07105423 [K], tailcorrection: yes
             Na - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  37.50184 [K], p_1: 2.90950 [A], shift/k_B:  -0.02068726 [K], tailcorrection: yes
             Na - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.33000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Na - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.33000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Na -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  20.62126 [K], p_1: 2.69000 [A], shift/k_B:  -0.00710545 [K], tailcorrection: yes
             Na -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  20.62126 [K], p_1: 2.69000 [A], shift/k_B:  -0.00710545 [K], tailcorrection: yes
             Na -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  20.62126 [K], p_1: 2.69000 [A], shift/k_B:  -0.00710545 [K], tailcorrection: yes
             Na -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  20.62126 [K], p_1: 2.69000 [A], shift/k_B:  -0.00710545 [K], tailcorrection: yes
             Cl -      Cl [LENNARD_JONES] p_0/k_B: 114.21000 [K], p_1: 3.52000 [A], shift/k_B:  -0.19750241 [K], tailcorrection: yes
             Cl -      Kr [LENNARD_JONES] p_0/k_B: 137.35899 [K], p_1: 3.59000 [A], shift/k_B:  -0.26730834 [K], tailcorrection: yes
             Cl -      Xe [LENNARD_JONES] p_0/k_B: 162.00450 [K], p_1: 3.74500 [A], shift/k_B:  -0.40622410 [K], tailcorrection: yes
             Cl - O_tip4p_2005 [LENNARD_JONES] p_0/k_B: 103.17157 [K], p_1: 3.33950 [A], shift/k_B:  -0.13011036 [K], tailcorrection: yes
             Cl - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.76000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Cl - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.76000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Cl -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  56.73128 [K], p_1: 3.12000 [A], shift/k_B:  -0.04758381 [K], tailcorrection: yes
             Cl -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  56.73128 [K], p_1: 3.12000 [A], shift/k_B:  -0.04758381 [K], tailcorrection: yes
             Cl -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  56.73128 [K], p_1: 3.12000 [A], shift/k_B:  -0.04758381 [K], tailcorrection: yes
             Cl -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  56.73128 [K], p_1: 3.12000 [A], shift/k_B:  -0.04758381 [K], tailcorrection: yes
             Kr -      Kr [LENNARD_JONES] p_0/k_B: 165.20000 [K], p_1: 3.66000 [A], shift/k_B:  -0.36096019 [K], tailcorrection: yes
             Kr -      Xe [LENNARD_JONES] p_0/k_B: 194.84086 [K], p_1: 3.81500 [A], shift/k_B:  -0.54593749 [K], tailcorrection: yes
             Kr - O_tip4p_2005 [LENNARD_JONES] p_0/k_B: 124.08320 [K], p_1: 3.40950 [A], shift/k_B:  -0.17721569 [K], tailcorrection: yes
             Kr - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.83000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Kr - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.83000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Kr -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  68.23002 [K], p_1: 3.19000 [A], shift/k_B:  -0.06537560 [K], tailcorrection: yes
             Kr -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  68.23002 [K], p_1: 3.19000 [A], shift/k_B:  -0.06537560 [K], tailcorrection: yes
             Kr -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  68.23002 [K], p_1: 3.19000 [A], shift/k_B:  -0.06537560 [K], tailcorrection: yes
             Kr -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  68.23002 [K], p_1: 3.19000 [A], shift/k_B:  -0.06537560 [K], tailcorrection: yes
             Xe -      Xe [LENNARD_JONES] p_0/k_B: 229.80000 [K], p_1: 3.97000 [A], shift/k_B:  -0.81753520 [K], tailcorrection: yes
             Xe - O_tip4p_2005 [LENNARD_JONES] p_0/k_B: 146.34671 [K], p_1: 3.56450 [A], shift/k_B:  -0.27288035 [K], tailcorrection: yes
             Xe - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.98500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Xe - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.98500 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
             Xe -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  80.47213 [K], p_1: 3.34500 [A], shift/k_B:  -0.10249061 [K], tailcorrection: yes
             Xe -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  80.47213 [K], p_1: 3.34500 [A], shift/k_B:  -0.10249061 [K], tailcorrection: yes
             Xe -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  80.47213 [K], p_1: 3.34500 [A], shift/k_B:  -0.10249061 [K], tailcorrection: yes
             Xe -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  80.47213 [K], p_1: 3.34500 [A], shift/k_B:  -0.10249061 [K], tailcorrection: yes
        O_tip4p_2005 - O_tip4p_2005 [LENNARD_JONES] p_0/k_B:  93.20000 [K], p_1: 3.15900 [A], shift/k_B:  -0.08422009 [K], tailcorrection: yes
        O_tip4p_2005 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.57950 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        O_tip4p_2005 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 1.57950 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        O_tip4p_2005 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  51.24818 [K], p_1: 2.93950 [A], shift/k_B:  -0.03006462 [K], tailcorrection: yes
        O_tip4p_2005 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  51.24818 [K], p_1: 2.93950 [A], shift/k_B:  -0.03006462 [K], tailcorrection: yes
        O_tip4p_2005 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  51.24818 [K], p_1: 2.93950 [A], shift/k_B:  -0.03006462 [K], tailcorrection: yes
        O_tip4p_2005 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  51.24818 [K], p_1: 2.93950 [A], shift/k_B:  -0.03006462 [K], tailcorrection: yes
        H_tip4p_2005 - H_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 0.00000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        H_tip4p_2005 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 0.00000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        H_tip4p_2005 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        H_tip4p_2005 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        H_tip4p_2005 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        H_tip4p_2005 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        M_tip4p_2005 - M_tip4p_2005 [LENNARD_JONES] p_0/k_B:   0.00000 [K], p_1: 0.00000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        M_tip4p_2005 -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        M_tip4p_2005 -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        M_tip4p_2005 -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
        M_tip4p_2005 -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:   0.00000 [K], p_1: 1.36000 [A], shift/k_B:  -0.00000000 [K], tailcorrection: yes
         Mof_Zn -  Mof_Zn [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
         Mof_Zn -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
         Mof_Zn -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
         Mof_Zn -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
          Mof_O -   Mof_O [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
          Mof_O -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
          Mof_O -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
          Mof_C -   Mof_C [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
          Mof_C -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes
          Mof_H -   Mof_H [LENNARD_JONES_CONTINUOUS_FRACTIONAL] p_0/k_B:  28.18000 [K], p_1: 2.72000 [A], shift/k_B:  -0.01037802 [K], tailcorrection: yes


        MoleculeDefinitions:
        ===========================================================================
        Component 0 [CO2] (Adsorbate molecule)

            MoleculeDefinitions: Hotpot
            Component contains (at least some) atoms which are charged
            Component contains no atoms with point dipoles (polarization)
            Component has a net charge of 0.000000

            Ideal chain Rosenbluth weight: 1
            Ideal chain total energy: 0.000000

            Critical temparure [K]: 304.128200
            Critical pressure [Pa]: 7377300.000000
            Acentric factor [-]: 0.223940

            RXMC partition factor ln(q/V) [ln(A^(-3))]:       0.0000000000

            Fluid is a vapour

            MolFraction:           0.3333333333 [-]
            Compressibility:       0.9973938375 [-]

            Density of the bulk fluid phase:       1.9681366726 [kg/m^3]

            Binary mixture EOS parameters:  (0): 0.000000 (1): 0.000000 (2): 0.000000

            Amount of excess molecules:       0.1547812161 [-]

            Conversion factor molecules/unit cell -> mol/kg:       7.4378371758 [-]
            Conversion factor molecules/unit cell -> mg/g:     327.2559103289 [-]
            Conversion factor molecules/unit cell -> cm^3 STP/gr:     166.7115020720 [-]
            Conversion factor molecules/unit cell -> cm^3 STP/cm^3:       2.1592046669 [-]
            Conversion factor mol/kg -> cm^3 STP/gr:      22.4139757476 [-]
            Conversion factor mol/kg -> cm^3 STP/cm^3:       0.2903000719 [-]

            Partial pressure:     33775.00000000000000 [Pa]
                                    253.31250000000000 [Torr]
                                      0.33775000000000 [bar]
                                      0.33333333333333 [atm]

            Fugacity coefficient:       0.9937963049 [-]

            Partial fugacity:     33565.47019664884283 [Pa]
                                    251.74102647486632 [Torr]
                                      0.33565470196649 [bar]
                                      0.33126543495336 [atm]

            Molecule contains 3 number of atoms
                atom:    0  is of type:   13 [     O_co2] (group: 0)
                atom:    1  is of type:   12 [     C_co2] (group: 0)
                atom:    2  is of type:   13 [     O_co2] (group: 0)

            Molecule contains 0 chirality centers

            Molecule contains 1 number of groups

                group: 0 containing: 3 elements
                -------------------------------------------------
                the group is rigid and linear
                Mass: 43.998800 [a.u.]
                Mass: 4.238536 [kg/m^3]

                Rotational Degrees of freedom: 2
                Diagonalized inertia vector:      42.2448477588
                                                  42.2448477588
                                                   0.0000000000
                number of atoms: 3
                    element: 0 atom: 0 [     O_co2] Charge: -0.350000 Anisotropy: 0.000000 Position:  0.000000 -0.000000  1.149000 Connectivity: 1 (1 )
                    element: 1 atom: 1 [     C_co2] Charge:  0.700000 Anisotropy: 0.000000 Position:  0.000000 -0.000000  0.000000 Connectivity: 2 (0 2 )
                    element: 2 atom: 2 [     O_co2] Charge: -0.350000 Anisotropy: 0.000000 Position:  0.000000 -0.000000 -1.149000 Connectivity: 1 (1 )
                number of permanent dipoles: 0
                number of polarizabilities: 0

                Dipole:           0.0000000000 [D]
                Quadrupole:       2.2194194468       2.2194194468      -4.4388388936 [D Angstrom]
                Quadrupole tensor [D Angstrom]
                               2.2194194468       0.0000000000       0.0000000000
                               0.0000000000       2.2194194468       0.0000000000
                               0.0000000000       0.0000000000      -4.4388388936

            Starting bead for growth               : 0

            Degrees of freedom                     : 0
            Translational degrees of freedom       : 0
            Rotational degrees of freedom          : 0
            Vibrational degrees of freedom         : 0
            Constraint degrees of freedom          : 0

            Number of atoms                        : 3

            Number of constraint bonds             : 0
            Number of constraint bends             : 0
            Number of constraint inversion bends   : 0
            Number of constraint torsions          : 0
            Number of constraint improper torsions : 0
            Number of constraint improper torsions : 0

            Number of bonds                        : 2
            Number of Urey-Bradleys                : 0
            Number of bends                        : 0
            Number of inversion bends              : 0
            Number of torsions                     : 0
            Number of improper torsions            : 0
            Number of improper torsions            : 0

            Number of bond/bond cross terms        : 0
            Number of bond/bend cross terms        : 0
            Number of bend/bend cross terms        : 0
            Number of stretch/torsion cross terms  : 0
            Number of bend/torsion cross terms     : 0

            Number of charges                      : 3
            Number of bond-dipoles                 : 0

            Number of intra Van der Waals                             : 0
            Number of intra charge-charge Coulomb                     : 0
            Number of intra charge-bonddipole Coulomb                 : 0
            Number of intra bonddipole-bonddipole Coulomb             : 0

            Number of excluded intra charge-charge Coulomb                     : 3
            Number of excluded intra charge-bonddipole Coulomb                 : 0
            Number of excluded intra bonddipole-bonddipole Coulomb             : 0

            Number of cbmc-config moves                               : 0

            Particle Moves:             
                ProbabilityTranslationMove:                  20.000000
                    TranslationDirection:      XYZ
                Percentage of random translation moves:            0.000000
                Percentage of rotation moves:                      20.000000
                Percentage of random rotation moves:               0.000000
                Percentage of partial reinsertion moves:           0.000000
                Percentage of reinsertion moves:                   20.000000
                Percentage of reinsertion-in-place moves:          0.000000
                Percentage of reinsertion-in-plane moves:          0.000000
                Percentage of identity-change moves:               20.000000
                    move 0    component 0 => 0
                    move 1    component 0 => 1
                    move 2    component 0 => 2
                Percentage of swap (insert/delete) moves:          20.000000
                Percentage of CF swap lambda moves:                0.000000
                Percentage of CB/CFMC swap lambda moves:           0.000000
                Percentage of Widom insertion moves:               0.000000
                Percentage of CF-Widom insertion moves:            0.000000
                Percentage of Gibbs Widom insertion moves:         0.000000
                Percentage of surface-area moves:                  0.000000
                Percentage of Gibbs particle-transfer moves:       0.000000
                Percentage of Gibbs identity-change moves:         0.000000
                Percentage of CF Gibbs lambda-transfer moves:      0.000000
                Percentage of CB/CFMC Gibbs lambda-transfer moves: 0.000000
                Percentage of exchange frac./int. particle moves:  0.000000
                Percentage of fractional mol. to other box moves:  0.000000
                Percentage of lambda-change moves:                 0.000000
                Percentage of fractional to integer moves:         0.000000

            System Moves:
                Percentage of parallel-tempering moves:            0.000000
                Percentage of hyper-parallel-tempering moves:      0.000000
                Percentage of parallel-mol-fraction moves:         0.000000
                       Component A: 0 B: 1
                Percentage of chiral inversion moves:              0.000000
                Percentage of Hybrid-NVE moves:                    0.000000
                Percentage of Hybrid-NPH moves:                    0.000000
                Percentage of Hybrid-NPHPR moves:                  0.000000
                Percentage of volume-change moves:                 0.000000
                Percentage of box-shape-change moves:              0.000000
                Percentage of Gibbs volume-change moves:           0.000000
                Percentage of framework-change moves:              0.000000
                Percentage of framework-shift moves:               0.000000
                Percentage of reactive MC moves:                   0.000000

            Moves are restricted: No
            No biased sampling used for this component

            Number of Bonds: 2
            --------------------------------------------
            Bond interaction 0: A=0 B=1 Type:RIGID_BOND
                r_0=1.1490000000       [A]
            Bond interaction 1: A=1 B=2 Type:RIGID_BOND
                r_0=1.1490000000       [A]


            number of identity changes: 3
            --------------------------------------------
            0 (CO2) to 0 (CO2)
            0 (CO2) to 1 (O2)
            0 (CO2) to 2 (N2)

            number of identity-config changes: 1
            --------------------------------------------
            nr fixed 1: 0 

            Number of pockets blocked in a unitcell: 0
                Pockets are NOT blocked for this component

        Component 1 [O2] (Adsorbate molecule)

            MoleculeDefinitions: Hotpot
            Component contains (at least some) atoms which are charged
            Component contains no atoms with point dipoles (polarization)
            Component has a net charge of 0.000000

            Ideal chain Rosenbluth weight: 1
            Ideal chain total energy: 0.000000

            Critical temparure [K]: 154.581000
            Critical pressure [Pa]: 5043000.000000
            Acentric factor [-]: 0.022200

            RXMC partition factor ln(q/V) [ln(A^(-3))]:       0.0000000000

            Fluid is a vapour

            MolFraction:           0.3333333333 [-]
            Compressibility:       0.9973938375 [-]

            Density of the bulk fluid phase:       1.4313574861 [kg/m^3]

            Binary mixture EOS parameters:  (0): 0.000000 (1): 0.000000 (2): 0.000000

            Amount of excess molecules:       0.1547812161 [-]

            Conversion factor molecules/unit cell -> mol/kg:       7.4378371758 [-]
            Conversion factor molecules/unit cell -> mg/g:     238.0018642197 [-]
            Conversion factor molecules/unit cell -> cm^3 STP/gr:     166.7115020720 [-]
            Conversion factor molecules/unit cell -> cm^3 STP/cm^3:       2.1592046669 [-]
            Conversion factor mol/kg -> cm^3 STP/gr:      22.4139757476 [-]
            Conversion factor mol/kg -> cm^3 STP/cm^3:       0.2903000719 [-]

            Partial pressure:     33775.00000000000000 [Pa]
                                    253.31250000000000 [Torr]
                                      0.33775000000000 [bar]
                                      0.33333333333333 [atm]

            Fugacity coefficient:       0.9988672188 [-]

            Partial fugacity:     33736.74031544724130 [Pa]
                                    253.02555236585431 [Torr]
                                      0.33736740315447 [bar]
                                      0.33295573960471 [atm]

            Molecule contains 3 number of atoms
                atom:    0  is of type:   14 [      O_o2] (group: 0)
                atom:    1  is of type:   15 [     O_com] (group: 0)
                atom:    2  is of type:   14 [      O_o2] (group: 0)

            Molecule contains 0 chirality centers

            Molecule contains 1 number of groups

                group: 0 containing: 3 elements
                -------------------------------------------------
                the group is rigid and linear
                Mass: 31.998800 [a.u.]
                Mass: 3.082540 [kg/m^3]

                Rotational Degrees of freedom: 2
                Diagonalized inertia vector:      11.5195680000
                                                  11.5195680000
                                                   0.0000000000
                number of atoms: 3
                    element: 0 atom: 0 [      O_o2] Charge: -0.113000 Anisotropy: 0.000000 Position:  0.000000 -0.000000  0.600000 Connectivity: 1 (1 )
                    element: 1 atom: 1 [     O_com] Charge:  0.226000 Anisotropy: 0.000000 Position:  0.000000 -0.000000  0.000000 Connectivity: 2 (0 2 )
                    element: 2 atom: 2 [      O_o2] Charge: -0.113000 Anisotropy: 0.000000 Position:  0.000000 -0.000000 -0.600000 Connectivity: 1 (1 )
                number of permanent dipoles: 0
                number of polarizabilities: 0

                Dipole:           0.0000000000 [D]
                Quadrupole:       0.1953944526       0.1953944526      -0.3907889052 [D Angstrom]
                Quadrupole tensor [D Angstrom]
                               0.1953944526       0.0000000000       0.0000000000
                               0.0000000000       0.1953944526       0.0000000000
                               0.0000000000       0.0000000000      -0.3907889052

            Starting bead for growth               : 0

            Degrees of freedom                     : 0
            Translational degrees of freedom       : 0
            Rotational degrees of freedom          : 0
            Vibrational degrees of freedom         : 0
            Constraint degrees of freedom          : 0

            Number of atoms                        : 3

            Number of constraint bonds             : 0
            Number of constraint bends             : 0
            Number of constraint inversion bends   : 0
            Number of constraint torsions          : 0
            Number of constraint improper torsions : 0
            Number of constraint improper torsions : 0

            Number of bonds                        : 2
            Number of Urey-Bradleys                : 0
            Number of bends                        : 0
            Number of inversion bends              : 0
            Number of torsions                     : 0
            Number of improper torsions            : 0
            Number of improper torsions            : 0

            Number of bond/bond cross terms        : 0
            Number of bond/bend cross terms        : 0
            Number of bend/bend cross terms        : 0
            Number of stretch/torsion cross terms  : 0
            Number of bend/torsion cross terms     : 0

            Number of charges                      : 3
            Number of bond-dipoles                 : 0

            Number of intra Van der Waals                             : 0
            Number of intra charge-charge Coulomb                     : 0
            Number of intra charge-bonddipole Coulomb                 : 0
            Number of intra bonddipole-bonddipole Coulomb             : 0

            Number of excluded intra charge-charge Coulomb                     : 3
            Number of excluded intra charge-bonddipole Coulomb                 : 0
            Number of excluded intra bonddipole-bonddipole Coulomb             : 0

            Number of cbmc-config moves                               : 0

            Particle Moves:             
                ProbabilityTranslationMove:                  20.000000
                    TranslationDirection:      XYZ
                Percentage of random translation moves:            0.000000
                Percentage of rotation moves:                      20.000000
                Percentage of random rotation moves:               0.000000
                Percentage of partial reinsertion moves:           0.000000
                Percentage of reinsertion moves:                   20.000000
                Percentage of reinsertion-in-place moves:          0.000000
                Percentage of reinsertion-in-plane moves:          0.000000
                Percentage of identity-change moves:               20.000000
                    move 0    component 1 => 0
                    move 1    component 1 => 1
                    move 2    component 1 => 2
                Percentage of swap (insert/delete) moves:          20.000000
                Percentage of CF swap lambda moves:                0.000000
                Percentage of CB/CFMC swap lambda moves:           0.000000
                Percentage of Widom insertion moves:               0.000000
                Percentage of CF-Widom insertion moves:            0.000000
                Percentage of Gibbs Widom insertion moves:         0.000000
                Percentage of surface-area moves:                  0.000000
                Percentage of Gibbs particle-transfer moves:       0.000000
                Percentage of Gibbs identity-change moves:         0.000000
                Percentage of CF Gibbs lambda-transfer moves:      0.000000
                Percentage of CB/CFMC Gibbs lambda-transfer moves: 0.000000
                Percentage of exchange frac./int. particle moves:  0.000000
                Percentage of fractional mol. to other box moves:  0.000000
                Percentage of lambda-change moves:                 0.000000
                Percentage of fractional to integer moves:         0.000000

            System Moves:
                Percentage of parallel-tempering moves:            0.000000
                Percentage of hyper-parallel-tempering moves:      0.000000
                Percentage of parallel-mol-fraction moves:         0.000000
                       Component A: 0 B: 1
                Percentage of chiral inversion moves:              0.000000
                Percentage of Hybrid-NVE moves:                    0.000000
                Percentage of Hybrid-NPH moves:                    0.000000
                Percentage of Hybrid-NPHPR moves:                  0.000000
                Percentage of volume-change moves:                 0.000000
                Percentage of box-shape-change moves:              0.000000
                Percentage of Gibbs volume-change moves:           0.000000
                Percentage of framework-change moves:              0.000000
                Percentage of framework-shift moves:               0.000000
                Percentage of reactive MC moves:                   0.000000

            Moves are restricted: No
            No biased sampling used for this component

            Number of Bonds: 2
            --------------------------------------------
            Bond interaction 0: A=0 B=1 Type:RIGID_BOND
                r_0=0.6000000000       [A]
            Bond interaction 1: A=1 B=2 Type:RIGID_BOND
                r_0=0.6000000000       [A]


            number of identity changes: 3
            --------------------------------------------
            1 (O2) to 0 (CO2)
            1 (O2) to 1 (O2)
            1 (O2) to 2 (N2)

            number of identity-config changes: 1
            --------------------------------------------
            nr fixed 1: 0 

            Number of pockets blocked in a unitcell: 0
                Pockets are NOT blocked for this component

        Component 2 [N2] (Adsorbate molecule)

            MoleculeDefinitions: Hotpot
            Component contains (at least some) atoms which are charged
            Component contains no atoms with point dipoles (polarization)
            Component has a net charge of 0.000000

            Ideal chain Rosenbluth weight: 1
            Ideal chain total energy: 0.000000

            Critical temparure [K]: 126.192000
            Critical pressure [Pa]: 3395800.000000
            Acentric factor [-]: 0.037200

            RXMC partition factor ln(q/V) [ln(A^(-3))]:       0.0000000000

            Fluid is a vapour

            MolFraction:           0.3333333333 [-]
            Compressibility:       0.9973938375 [-]

            Density of the bulk fluid phase:       1.2530877505 [kg/m^3]

            Binary mixture EOS parameters:  (0): 0.000000 (1): 0.000000 (2): 0.000000

            Amount of excess molecules:       0.1547812161 [-]

            Conversion factor molecules/unit cell -> mol/kg:       7.4378371758 [-]
            Conversion factor molecules/unit cell -> mg/g:     208.3597029664 [-]
            Conversion factor molecules/unit cell -> cm^3 STP/gr:     166.7115020720 [-]
            Conversion factor molecules/unit cell -> cm^3 STP/cm^3:       2.1592046669 [-]
            Conversion factor mol/kg -> cm^3 STP/gr:      22.4139757476 [-]
            Conversion factor mol/kg -> cm^3 STP/cm^3:       0.2903000719 [-]

            Partial pressure:     33775.00000000000000 [Pa]
                                    253.31250000000000 [Torr]
                                      0.33775000000000 [bar]
                                      0.33333333333333 [atm]

            Fugacity coefficient:       0.9995350781 [-]

            Partial fugacity:     33759.29726429879520 [Pa]
                                    253.19472948224094 [Torr]
                                      0.33759297264299 [bar]
                                      0.33317835938119 [atm]

            Molecule contains 3 number of atoms
                atom:    0  is of type:   16 [      N_n2] (group: 0)
                atom:    1  is of type:   17 [     N_com] (group: 0)
                atom:    2  is of type:   16 [      N_n2] (group: 0)

            Molecule contains 0 chirality centers

            Molecule contains 1 number of groups

                group: 0 containing: 3 elements
                -------------------------------------------------
                the group is rigid and linear
                Mass: 28.013480 [a.u.]
                Mass: 2.698622 [kg/m^3]

                Rotational Degrees of freedom: 2
                Diagonalized inertia vector:       8.4740777000
                                                   8.4740777000
                                                   0.0000000000
                number of atoms: 3
                    element: 0 atom: 0 [      N_n2] Charge: -0.482000 Anisotropy: 0.000000 Position:  0.000000 -0.000000  0.550000 Connectivity: 1 (1 )
                    element: 1 atom: 1 [     N_com] Charge:  0.964000 Anisotropy: 0.000000 Position:  0.000000 -0.000000  0.000000 Connectivity: 2 (0 2 )
                    element: 2 atom: 2 [      N_n2] Charge: -0.482000 Anisotropy: 0.000000 Position:  0.000000 -0.000000 -0.550000 Connectivity: 1 (1 )
                number of permanent dipoles: 0
                number of polarizabilities: 0

                Dipole:           0.0000000000 [D]
                Quadrupole:       0.7003315673       0.7003315673      -1.4006631347 [D Angstrom]
                Quadrupole tensor [D Angstrom]
                               0.7003315673       0.0000000000       0.0000000000
                               0.0000000000       0.7003315673       0.0000000000
                               0.0000000000       0.0000000000      -1.4006631347

            Starting bead for growth               : 0

            Degrees of freedom                     : 0
            Translational degrees of freedom       : 0
            Rotational degrees of freedom          : 0
            Vibrational degrees of freedom         : 0
            Constraint degrees of freedom          : 0

            Number of atoms                        : 3

            Number of constraint bonds             : 0
            Number of constraint bends             : 0
            Number of constraint inversion bends   : 0
            Number of constraint torsions          : 0
            Number of constraint improper torsions : 0
            Number of constraint improper torsions : 0

            Number of bonds                        : 2
            Number of Urey-Bradleys                : 0
            Number of bends                        : 0
            Number of inversion bends              : 0
            Number of torsions                     : 0
            Number of improper torsions            : 0
            Number of improper torsions            : 0

            Number of bond/bond cross terms        : 0
            Number of bond/bend cross terms        : 0
            Number of bend/bend cross terms        : 0
            Number of stretch/torsion cross terms  : 0
            Number of bend/torsion cross terms     : 0

            Number of charges                      : 3
            Number of bond-dipoles                 : 0

            Number of intra Van der Waals                             : 0
            Number of intra charge-charge Coulomb                     : 0
            Number of intra charge-bonddipole Coulomb                 : 0
            Number of intra bonddipole-bonddipole Coulomb             : 0

            Number of excluded intra charge-charge Coulomb                     : 3
            Number of excluded intra charge-bonddipole Coulomb                 : 0
            Number of excluded intra bonddipole-bonddipole Coulomb             : 0

            Number of cbmc-config moves                               : 0

            Particle Moves:             
                ProbabilityTranslationMove:                  20.000000
                    TranslationDirection:      XYZ
                Percentage of random translation moves:            0.000000
                Percentage of rotation moves:                      20.000000
                Percentage of random rotation moves:               0.000000
                Percentage of partial reinsertion moves:           0.000000
                Percentage of reinsertion moves:                   20.000000
                Percentage of reinsertion-in-place moves:          0.000000
                Percentage of reinsertion-in-plane moves:          0.000000
                Percentage of identity-change moves:               20.000000
                    move 0    component 2 => 0
                    move 1    component 2 => 1
                    move 2    component 2 => 2
                Percentage of swap (insert/delete) moves:          20.000000
                Percentage of CF swap lambda moves:                0.000000
                Percentage of CB/CFMC swap lambda moves:           0.000000
                Percentage of Widom insertion moves:               0.000000
                Percentage of CF-Widom insertion moves:            0.000000
                Percentage of Gibbs Widom insertion moves:         0.000000
                Percentage of surface-area moves:                  0.000000
                Percentage of Gibbs particle-transfer moves:       0.000000
                Percentage of Gibbs identity-change moves:         0.000000
                Percentage of CF Gibbs lambda-transfer moves:      0.000000
                Percentage of CB/CFMC Gibbs lambda-transfer moves: 0.000000
                Percentage of exchange frac./int. particle moves:  0.000000
                Percentage of fractional mol. to other box moves:  0.000000
                Percentage of lambda-change moves:                 0.000000
                Percentage of fractional to integer moves:         0.000000

            System Moves:
                Percentage of parallel-tempering moves:            0.000000
                Percentage of hyper-parallel-tempering moves:      0.000000
                Percentage of parallel-mol-fraction moves:         0.000000
                       Component A: 0 B: 1
                Percentage of chiral inversion moves:              0.000000
                Percentage of Hybrid-NVE moves:                    0.000000
                Percentage of Hybrid-NPH moves:                    0.000000
                Percentage of Hybrid-NPHPR moves:                  0.000000
                Percentage of volume-change moves:                 0.000000
                Percentage of box-shape-change moves:              0.000000
                Percentage of Gibbs volume-change moves:           0.000000
                Percentage of framework-change moves:              0.000000
                Percentage of framework-shift moves:               0.000000
                Percentage of reactive MC moves:                   0.000000

            Moves are restricted: No
            No biased sampling used for this component

            Number of Bonds: 2
            --------------------------------------------
            Bond interaction 0: A=0 B=1 Type:RIGID_BOND
                r_0=0.5500000000       [A]
            Bond interaction 1: A=1 B=2 Type:RIGID_BOND
                r_0=0.5500000000       [A]


            number of identity changes: 3
            --------------------------------------------
            2 (N2) to 0 (CO2)
            2 (N2) to 1 (O2)
            2 (N2) to 2 (N2)

            number of identity-config changes: 1
            --------------------------------------------
            nr fixed 1: 0 

            Number of pockets blocked in a unitcell: 0
                Pockets are NOT blocked for this component



        Framework Status
        ===========================================================================
        Lowenstein's rule obeyed by framework
            Framework is modelled as: rigid

            Number of charges:                               0
            Number of bonddipoles:                           0


        System Properties
        ===========================================================================
        Unit cell size: 25.832000 25.832000 25.832000
        Cell angles (radians)  alpha: 1.570796 beta: 1.570796 gamma: 1.570796
        Cell angles (degrees)  alpha: 90.000000 beta: 90.000000 gamma: 90.000000
        Number of unitcells [a]: 1
        Number of unitcells [b]: 1
        Number of unitcells [c]: 1

        TRICLINIC Boundary conditions: alpha!=90 or beta!=90 or gamma!=90

        Cartesian axis A is collinear with crystallographic axis a
        Cartesian axis B is collinear with (axb)xA
        Cartesian axis C is collinear with (axb)

        lengths of cell vectors:
         25.83200  25.83200  25.83200
        cosines of cell angles:
          0.00000   0.00000   0.00000
        perpendicular cell widths:
         25.83200  25.83200  25.83200
        volume of the cell: 17237.492730368002 (A^3)

        Orthogonalization matrix Box
        Transforms fractional coordinates abc into orthonormal Cartesian coordinates xyz
        Deorthogonalization matrix InverseBox
        Transforms orthonormal Cartesian coordinates xyz into fractional coordinates xyz

        Box[0]:
               25.832000000000     0.000000000000     0.000000000000
                0.000000000000    25.832000000000     0.000000000000
                0.000000000000     0.000000000000    25.832000000000

        Inverse box[0]:
                0.038711675441    -0.000000000000    -0.000000000000
                0.000000000000     0.038711675441    -0.000000000000
                0.000000000000     0.000000000000     0.038711675441


        Unitcell box[0]:
               25.832000000000     0.000000000000     0.000000000000
                0.000000000000    25.832000000000     0.000000000000
                0.000000000000     0.000000000000    25.832000000000

        Unitcell inverse box[0]:
                0.038711675441    -0.000000000000    -0.000000000000
                0.000000000000     0.038711675441    -0.000000000000
                0.000000000000     0.000000000000     0.038711675441

        lengths of cell vectors (inverse box):
          0.03871   0.03871   0.03871
        cosines of cell angles (inverse box):
         -0.00000  -0.00000  -0.00000
        perpendicular cell widths (inverse):
          0.03871   0.03871   0.03871
        volume of the cell: 17237.492730368002 (A^3)

        No replicas are used
        Framework is simulated as 'rigid'
        Number of framework atoms: 7
        Number of framework atoms in the unit cell: 7
        Framework Mass:   134.447686386433 [g/mol]
        Framework Density:    12.951743821137 [kg/m^3]   77.2096803187236 [cm^3/g]
        Helium void fraction:    1.00000000
        Available pore volume: 17237.49273037 [A^3]   77.20968032 [cm^3/g]
        Conversion factor from molecule/unit cell -> kmol/m^3: 0.096333, kmol/m^3 accesible pore volume: 0.096333

        Number Of Frameworks (per system): 1
        ----------------------------------------------------------------------
        Framework name: streamed
        Space group: 1
            Identifier: 1
            short international Hermann-Mauguin symbol: P 1
            long international Hermann-Mauguin symbol: P 1
            Hall symbol: P 1
            Number of lattice translations: 1 [ (0,0,0) ]
            acentric/centric: acentric
            chiral: no
            enantiomorphic: no
            number of operators: 1
                'x,y,z'
        Framework is simulated as 'rigid'
        Shift: 0.000000 0.000000 0.000000
        Number of framework atoms: 7
        Number of asymmetric atoms: 7
        Number of free framework atoms: 0
        Number of fixed framework atoms: 7
        Number of framework atoms in the unit cell: 7
        Framework Mass:   134.447686386433 [g/mol]
        Framework Density:    12.951743821137 [kg/m^3]
        Framework has net charge: 0.000000
                 largest charge : 0.000000
                 smallest charge: 0.000000

        Using FULL Host-guest interaction calculation (for testing purposes)

        Current Atom Status
        ===========================================================================
        Number of framework atoms        : 7
        Number of cations molecules      : 0
        Number of adsorbate molecules    : 0
        Component    0 :    0 molecules
        Component    1 :    0 molecules
        Component    2 :    0 molecules
        Pseudo Atoms    0 [    UNIT]:    0 atoms
        Pseudo Atoms    1 [      He]:    0 atoms
        Pseudo Atoms    2 [ CH4_sp3]:    0 atoms
        Pseudo Atoms    3 [ CH3_sp3]:    0 atoms
        Pseudo Atoms    4 [ CH2_sp3]:    0 atoms
        Pseudo Atoms    5 [CH3_sp3_ethane]:    0 atoms
        Pseudo Atoms    6 [CH2_sp2_ethene]:    0 atoms
        Pseudo Atoms    7 [I2_united_atom]:    0 atoms
        Pseudo Atoms    8 [  CH_sp3]:    0 atoms
        Pseudo Atoms    9 [   C_sp3]:    0 atoms
        Pseudo Atoms   10 [    H_h2]:    0 atoms
        Pseudo Atoms   11 [   H_com]:    0 atoms
        Pseudo Atoms   12 [   C_co2]:    0 atoms
        Pseudo Atoms   13 [   O_co2]:    0 atoms
        Pseudo Atoms   14 [    O_o2]:    0 atoms
        Pseudo Atoms   15 [   O_com]:    0 atoms
        Pseudo Atoms   16 [    N_n2]:    0 atoms
        Pseudo Atoms   17 [   N_com]:    0 atoms
        Pseudo Atoms   18 [      Ar]:    0 atoms
        Pseudo Atoms   19 [      Ow]:    0 atoms
        Pseudo Atoms   20 [      Hw]:    0 atoms
        Pseudo Atoms   21 [      Lw]:    0 atoms
        Pseudo Atoms   22 [  C_benz]:    0 atoms
        Pseudo Atoms   23 [  H_benz]:    0 atoms
        Pseudo Atoms   24 [   N_dmf]:    0 atoms
        Pseudo Atoms   25 [  Co_dmf]:    0 atoms
        Pseudo Atoms   26 [  Cm_dmf]:    0 atoms
        Pseudo Atoms   27 [   O_dmf]:    0 atoms
        Pseudo Atoms   28 [   H_dmf]:    0 atoms
        Pseudo Atoms   29 [      Na]:    0 atoms
        Pseudo Atoms   30 [      Cl]:    0 atoms
        Pseudo Atoms   31 [      Kr]:    0 atoms
        Pseudo Atoms   32 [      Xe]:    0 atoms
        Pseudo Atoms   33 [O_tip4p_2005]:    0 atoms
        Pseudo Atoms   34 [H_tip4p_2005]:    0 atoms
        Pseudo Atoms   35 [M_tip4p_2005]:    0 atoms
        Pseudo Atoms   36 [  Mof_Zn]:    1 atoms
        Pseudo Atoms   37 [   Mof_O]:    2 atoms
        Pseudo Atoms   38 [   Mof_C]:    3 atoms
        Pseudo Atoms   39 [   Mof_H]:    1 atoms


        Current (initial full energy) Energy Status
        ===========================================================================

        Internal energy:
        Host stretch energy:                                            0.00000000
        Host UreyBradley energy:                                        0.00000000
        Host bend energy:                                               0.00000000
        Host inversion-bend energy:                                     0.00000000
        Host torsion energy:                                            0.00000000
        Host improper torsion energy:                                   0.00000000
        Host out-of-plane energy:                                       0.00000000
        Host stretch/stretch energy:                                    0.00000000
        Host bend/bend energy:                                          0.00000000
        Host stretch/bend energy:                                       0.00000000
        Host stretch/torsion energy:                                    0.00000000
        Host bend/torsion energy:                                       0.00000000

        Adsorbate stretch energy:                                       0.00000000
        Adsorbate UreyBradley energy:                                   0.00000000
        Adsorbate bend energy:                                          0.00000000
        Adsorbate inversion-bend energy:                                0.00000000
        Adsorbate torsion energy:                                       0.00000000
        Adsorbate improper torsion energy:                              0.00000000
        Adsorbate out-of-plane energy:                                  0.00000000
        Adsorbate stretch/stretch energy:                               0.00000000
        Adsorbate bend/bend energy:                                     0.00000000
        Adsorbate stretch/bend energy:                                  0.00000000
        Adsorbate stretch/torsion energy:                               0.00000000
        Adsorbate bend/torsion energy:                                  0.00000000
        Adsorbate intra VDW energy:                                     0.00000000
        Adsorbate intra charge-charge Coulomb energy:                   0.00000000
        Adsorbate intra charge-bonddipole Coulomb energy:               0.00000000
        Adsorbate intra bonddipole-bonddipole Coulomb energy:           0.00000000

        Cation stretch energy:                                          0.00000000
        Cation UreyBradley energy:                                      0.00000000
        Cation bend energy:                                             0.00000000
        Cation inversion-bend energy:                                   0.00000000
        Cation torsion energy:                                          0.00000000
        Cation improper torsion energy:                                 0.00000000
        Cation out-of-plane energy:                                     0.00000000
        Cation stretch/stretch energy:                                  0.00000000
        Cation bend/bend energy:                                        0.00000000
        Cation stretch/bend energy:                                     0.00000000
        Cation stretch/torsion energy:                                  0.00000000
        Cation bend/torsion energy:                                     0.00000000
        Cation intra VDW energy:                                        0.00000000
        Cation intra charge-charge Coulomb energy:                      0.00000000
        Cation intra charge-bonddipole Coulomb energy:                  0.00000000
        Cation intra bonddipole-bonddipole Coulomb energy:              0.00000000

        Host/Host energy:                                             0.00000000
            Host/Host VDW energy:                                         0.00000000
            Host/Host Coulomb energy:                                     0.00000000
            Host/Host charge-charge Real energy:                          0.00000000
            Host/Host charge-charge Fourier energy:                       0.00000000
            Host/Host charge-bonddipole Real energy:                      0.00000000
            Host/Host charge-bonddipole Fourier energy:                   0.00000000
            Host/Host bondipole-bonddipole Real energy:                   0.00000000
            Host/Host bondipole-bonddipole Fourier energy:                0.00000000

        Host/Adsorbate energy:                                        0.00000000
            Host/Adsorbate VDW energy:                                    0.00000000
            Host/Adsorbate Coulomb energy:                                0.00000000
            Host/Adsorbate charge-charge Real energy:                     0.00000000
            Host/Adsorbate charge-charge Fourier energy:                  0.00000000
            Host/Adsorbate charge-bonddipole Real energy:                 0.00000000
            Host/Adsorbate charge-bonddipole Fourier energy:              0.00000000
            Host/Adsorbate bondipole-bonddipole Real energy:              0.00000000
            Host/Adsorbate bondipole-bonddipole Fourier energy:           0.00000000

        Host/Cation energy:                                           0.00000000
            Host/Cation VDW energy:                                       0.00000000
            Host/Cation Coulomb energy:                                   0.00000000
            Host/Cation charge-charge Real energy:                        0.00000000
            Host/Cation charge-charge Fourier energy:                     0.00000000
            Host/Cation charge-bonddipole Real energy:                    0.00000000
            Host/Cation charge-bonddipole Fourier energy:                 0.00000000
            Host/Cation bondipole-bonddipole Real energy:                 0.00000000
            Host/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Adsorbate/Adsorbate energy:                                        0.00000000
            Adsorbate/Adsorbate VDW energy:                                    0.00000000
            Adsorbate/Adsorbate Coulomb energy:                                0.00000000
            Adsorbate/Adsorbate charge-charge Real energy:                     0.00000000
            Adsorbate/Adsorbate charge-charge Fourier energy:                  0.00000000
            Adsorbate/Adsorbate charge-bonddipole Real energy:                 0.00000000
            Adsorbate/Adsorbate charge-bonddipole Fourier energy:              0.00000000
            Adsorbate/Adsorbate bondipole-bonddipole Real energy:              0.00000000
            Adsorbate/Adsorbate bondipole-bonddipole Fourier energy:           0.00000000

        Adsorbate/Cation energy:                                           0.00000000
            Adsorbate/Cation VDW energy:                                       0.00000000
            Adsorbate/Cation Coulomb energy:                                   0.00000000
            Adsorbate/Cation charge-charge Real energy:                        0.00000000
            Adsorbate/Cation charge-charge Fourier energy:                     0.00000000
            Adsorbate/Cation charge-bonddipole Real energy:                    0.00000000
            Adsorbate/Cation charge-bonddipole Fourier energy:                 0.00000000
            Adsorbate/Cation bondipole-bonddipole Real energy:                 0.00000000
            Adsorbate/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Cation/Cation energy:                                           0.00000000
            Cation/Cation VDW energy:                                       0.00000000
            Cation/Cation Coulomb energy:                                   0.00000000
            Cation/Cation charge-charge Real energy:                        0.00000000
            Cation/Cation charge-charge Fourier energy:                     0.00000000
            Cation/Cation charge-bonddipole Real energy:                    0.00000000
            Cation/Cation charge-bonddipole Fourier energy:                 0.00000000
            Cation/Cation bondipole-bonddipole Real energy:                 0.00000000
            Cation/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Polarization energy:
            Host polarization energy:                                     0.00000000
            Adsorbate polarization energy:                                0.00000000
            Cation polarization energy:                                   0.00000000
            Host back-polarization energy:                                     0.00000000
            Adsorbate back-polarization energy:                                0.00000000
            Cation back-polarization energy:                                   0.00000000

        Tail-correction energy:                                      -0.12958411

        Distance constraints energy:                                  0.00000000
        Angle constraints energy:                                     0.00000000
        Dihedral constraints energy:                                  0.00000000
        Inversion-bend constraints energy:                            0.00000000
        Out-of-plane distance constraints energy:                     0.00000000
        Exclusion constraints energy:                                 0.00000000

        ===================================================================
        Total energy:    -0.129584113858
            Total Van der Waals: 0.000000
            Total Coulomb: 0.000000

            Total Polarization: 0.000000







        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Starting simulation
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        [Init] Current cycle: 0 out of 10000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]
        Box-lengths:  25.83200  25.83200  25.83200 Box-angles:   90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0, density:   0.00000 [kg/m^3]
            absolute adsorption:   0.00000 [mol/uc],         0.0000 [mol/kg],              0.0000 [mg/g]
                                                             0.0000 [cm^3 STP/g],          0.0000 [cm^3 STP/cm^3]
            excess adsorption:    -0.15478 [mol/uc],        -1.1512 [mol/kg],            -50.6531 [mg/g]
                                                           -25.8038 [cm^3 STP/g],         -0.3342 [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0, density:   0.00000 [kg/m^3]
            absolute adsorption:   0.00000 [mol/uc],         0.0000 [mol/kg],              0.0000 [mg/g]
                                                             0.0000 [cm^3 STP/g],          0.0000 [cm^3 STP/cm^3]
            excess adsorption:    -0.15478 [mol/uc],        -1.1512 [mol/kg],            -36.8382 [mg/g]
                                                           -25.8038 [cm^3 STP/g],         -0.3342 [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0, density:   0.00000 [kg/m^3]
            absolute adsorption:   0.00000 [mol/uc],         0.0000 [mol/kg],              0.0000 [mg/g]
                                                             0.0000 [cm^3 STP/g],          0.0000 [cm^3 STP/cm^3]
            excess adsorption:    -0.15478 [mol/uc],        -1.1512 [mol/kg],            -32.2502 [mg/g]
                                                           -25.8038 [cm^3 STP/g],         -0.3342 [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction

        Current total potential energy:                -0.1295841139 [K]
            Current Host-Host energy:                     0.0000000000 [K]
            Current Host-Adsorbate energy:                0.0000000000 [K]
            Current Host-Cation energy:                   0.0000000000 [K]
            Current Adsorbate-Adsorbate energy:           0.0000000000 [K]
            Current Cation-Cation energy:                 0.0000000000 [K]
            Current Adsorbate-Cation energy:              0.0000000000 [K]

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 0 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 0 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.00000), density:   0.00000 (avg.   0.00000) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.00000) [mol/uc],   0.0000000000 (avg.   0.0000000000) [mol/kg],   0.0000000000 (avg.   0.0000000000) [mg/g]
                                   0.0000000000 (avg.   0.0000000000) [cm^3 STP/g],    0.0000000000 (avg.   0.0000000000) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.  -0.1547812161) [mol/uc],  -1.1512374831 (avg.  -1.1512374831) [mol/kg], -50.6530677727 (avg. -50.6530677727) [mg/g]
                                 -25.8038090266 (avg. -25.8038090266) [cm^3 STP/g],   -0.3342043241 (avg.  -0.3342043241) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.00000), density:   0.00000 (avg.   0.00000) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.00000) [mol/uc],   0.0000000000 (avg.   0.0000000000) [mol/kg],   0.0000000000 (avg.   0.0000000000) [mg/g]
                                   0.0000000000 (avg.   0.0000000000) [cm^3 STP/g],    0.0000000000 (avg.   0.0000000000) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.  -0.1547812161) [mol/uc],  -1.1512374831 (avg.  -1.1512374831) [mol/kg], -36.8382179752 (avg. -36.8382179752) [mg/g]
                                 -25.8038090266 (avg. -25.8038090266) [cm^3 STP/g],   -0.3342043241 (avg.  -0.3342043241) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.00000), density:   0.00000 (avg.   0.00000) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.00000) [mol/uc],   0.0000000000 (avg.   0.0000000000) [mol/kg],   0.0000000000 (avg.   0.0000000000) [mg/g]
                                   0.0000000000 (avg.   0.0000000000) [cm^3 STP/g],    0.0000000000 (avg.   0.0000000000) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.  -0.1547812161) [mol/uc],  -1.1512374831 (avg.  -1.1512374831) [mol/kg], -32.2501682089 (avg. -32.2501682089) [mg/g]
                                 -25.8038090266 (avg. -25.8038090266) [cm^3 STP/g],   -0.3342043241 (avg.  -0.3342043241) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841139 [K]  (avg.          -0.1295841139)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:                0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.0000000000 [K]  (avg.           0.0000000000)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 10000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 10000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15948), density:   0.00000 (avg.   0.67598) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15948) [mol/uc],   0.0000000000 (avg.   1.1862164079) [mol/kg],   0.0000000000 (avg.  52.1920984876) [mg/g]
                                   0.0000000000 (avg.  26.5878257979) [cm^3 STP/g],    0.0000000000 (avg.   0.3443587085) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0047028355) [mol/uc],  -1.1512374831 (avg.   0.0349789248) [mol/kg], -50.6530677727 (avg.   1.5390307149) [mg/g]
                                 -25.8038090266 (avg.   0.7840167713) [cm^3 STP/g],   -0.3342043241 (avg.   0.0101543844) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.16158), density:   0.00000 (avg.   0.49809) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.16158) [mol/uc],   0.0000000000 (avg.   1.2018343042) [mol/kg],   0.0000000000 (avg.  38.4572555324) [mg/g]
                                   0.0000000000 (avg.  26.9378849463) [cm^3 STP/g],    0.0000000000 (avg.   0.3488925849) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0068026255) [mol/uc],  -1.1512374831 (avg.   0.0505968210) [mol/kg], -36.8382179752 (avg.   1.6190375572) [mg/g]
                                 -25.8038090266 (avg.   1.1340759198) [cm^3 STP/g],   -0.3342043241 (avg.   0.0146882608) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15818), density:   0.00000 (avg.   0.42688) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15818) [mol/uc],   0.0000000000 (avg.   1.1765481864) [mol/kg],   0.0000000000 (avg.  32.9592090884) [mg/g]
                                   0.0000000000 (avg.  26.3711225155) [cm^3 STP/g],    0.0000000000 (avg.   0.3415520231) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0034029655) [mol/uc],  -1.1512374831 (avg.   0.0253107033) [mol/kg], -32.2501682089 (avg.   0.7090408795) [mg/g]
                                 -25.8038090266 (avg.   0.5673134890) [cm^3 STP/g],   -0.3342043241 (avg.   0.0073476990) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841138 [K]  (avg.          -5.4688621618)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.3569657310)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.0000000000 [K]  (avg.          -0.8629860736)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 20000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 20000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.16184), density:   0.00000 (avg.   0.68597) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.16184) [mol/uc],   0.0000000000 (avg.   1.2037537592) [mol/kg],   0.0000000000 (avg.  52.9637209007) [mg/g]
                                   0.0000000000 (avg.  26.9809075650) [cm^3 STP/g],    0.0000000000 (avg.   0.3494498028) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0070606918) [mol/uc],  -1.1512374831 (avg.   0.0525162761) [mol/kg], -50.6530677727 (avg.   2.3106531279) [mg/g]
                                 -25.8038090266 (avg.   1.1770985384) [cm^3 STP/g],   -0.3342043241 (avg.   0.0152454787) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15594), density:   0.00000 (avg.   0.48070) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15594) [mol/uc],   0.0000000000 (avg.   1.1598727139) [mol/kg],   0.0000000000 (avg.  37.1145349983) [mg/g]
                                   0.0000000000 (avg.  25.9973588802) [cm^3 STP/g],    0.0000000000 (avg.   0.3367111322) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0011609868) [mol/uc],  -1.1512374831 (avg.   0.0086352308) [mol/kg], -36.8382179752 (avg.   0.2763170231) [mg/g]
                                 -25.8038090266 (avg.   0.1935498536) [cm^3 STP/g],   -0.3342043241 (avg.   0.0025068081) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.16054), density:   0.00000 (avg.   0.43324) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.16054) [mol/uc],   0.0000000000 (avg.   1.1940850543) [mol/kg],   0.0000000000 (avg.  33.4504777874) [mg/g]
                                   0.0000000000 (avg.  26.7641934480) [cm^3 STP/g],    0.0000000000 (avg.   0.3466429771) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0057607568) [mol/uc],  -1.1512374831 (avg.   0.0428475712) [mol/kg], -32.2501682089 (avg.   1.2003095785) [mg/g]
                                 -25.8038090266 (avg.   0.9603844214) [cm^3 STP/g],   -0.3342043241 (avg.   0.0124386530) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841139 [K]  (avg.          -5.3875685282)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.2498592480)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:          -0.0000000001 [K]  (avg.          -0.8879928909)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 30000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 30000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15999), density:   0.00000 (avg.   0.67814) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15999) [mol/uc],   0.0000000000 (avg.   1.1900142810) [mol/kg],   0.0000000000 (avg.  52.3592003459) [mg/g]
                                   0.0000000000 (avg.  26.6729512332) [cm^3 STP/g],    0.0000000000 (avg.   0.3454612313) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0052134508) [mol/uc],  -1.1512374831 (avg.   0.0387767978) [mol/kg], -50.6530677727 (avg.   1.7061325732) [mg/g]
                                 -25.8038090266 (avg.   0.8691422066) [cm^3 STP/g],   -0.3342043241 (avg.   0.0112569072) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15606), density:   0.00000 (avg.   0.48107) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15606) [mol/uc],   0.0000000000 (avg.   1.1607597632) [mol/kg],   0.0000000000 (avg.  37.1429195119) [mg/g]
                                   0.0000000000 (avg.  26.0172411820) [cm^3 STP/g],    0.0000000000 (avg.   0.3369686427) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0012802485) [mol/uc],  -1.1512374831 (avg.   0.0095222801) [mol/kg], -36.8382179752 (avg.   0.3047015367) [mg/g]
                                 -25.8038090266 (avg.   0.2134321554) [cm^3 STP/g],   -0.3342043241 (avg.   0.0027643186) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15843), density:   0.00000 (avg.   0.42754) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15843) [mol/uc],   0.0000000000 (avg.   1.1783620578) [mol/kg],   0.0000000000 (avg.  33.0100219392) [mg/g]
                                   0.0000000000 (avg.  26.4117785857) [cm^3 STP/g],    0.0000000000 (avg.   0.3420785901) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0036468363) [mol/uc],  -1.1512374831 (avg.   0.0271245747) [mol/kg], -32.2501682089 (avg.   0.7598537303) [mg/g]
                                 -25.8038090266 (avg.   0.6079695591) [cm^3 STP/g],   -0.3342043241 (avg.   0.0078742660) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841139 [K]  (avg.          -5.3614655239)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.2676619861)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:          -0.0000000000 [K]  (avg.          -0.8452051425)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 40000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 40000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15990), density:   0.00000 (avg.   0.67773) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15990) [mol/uc],   0.0000000000 (avg.   1.1892804324) [mol/kg],   0.0000000000 (avg.  52.3269118888) [mg/g]
                                   0.0000000000 (avg.  26.6565027688) [cm^3 STP/g],    0.0000000000 (avg.   0.3452481950) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0051147865) [mol/uc],  -1.1512374831 (avg.   0.0380429493) [mol/kg], -50.6530677727 (avg.   1.6738441160) [mg/g]
                                 -25.8038090266 (avg.   0.8526937422) [cm^3 STP/g],   -0.3342043241 (avg.   0.0110438709) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15690), density:   0.00000 (avg.   0.48364) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15690) [mol/uc],   0.0000000000 (avg.   1.1669674787) [mol/kg],   0.0000000000 (avg.  37.3415589571) [mg/g]
                                   0.0000000000 (avg.  26.1563807656) [cm^3 STP/g],    0.0000000000 (avg.   0.3387707430) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0021148615) [mol/uc],  -1.1512374831 (avg.   0.0157299956) [mol/kg], -36.8382179752 (avg.   0.5033409819) [mg/g]
                                 -25.8038090266 (avg.   0.3525717390) [cm^3 STP/g],   -0.3342043241 (avg.   0.0045664188) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15825), density:   0.00000 (avg.   0.42705) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15825) [mol/uc],   0.0000000000 (avg.   1.1770083079) [mol/kg],   0.0000000000 (avg.  32.9720986920) [mg/g]
                                   0.0000000000 (avg.  26.3814356670) [cm^3 STP/g],    0.0000000000 (avg.   0.3416855964) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0034648278) [mol/uc],  -1.1512374831 (avg.   0.0257708247) [mol/kg], -32.2501682089 (avg.   0.7219304831) [mg/g]
                                 -25.8038090266 (avg.   0.5776266404) [cm^3 STP/g],   -0.3342043241 (avg.   0.0074812723) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841139 [K]  (avg.          -5.3564887636)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.2817886901)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:          -0.0000000000 [K]  (avg.          -0.8260202756)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 50000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 50000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.16024), density:   0.00000 (avg.   0.67917) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.16024) [mol/uc],   0.0000000000 (avg.   1.1918151927) [mol/kg],   0.0000000000 (avg.  52.4384383023) [mg/g]
                                   0.0000000000 (avg.  26.7133168257) [cm^3 STP/g],    0.0000000000 (avg.   0.3459840361) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0054555792) [mol/uc],  -1.1512374831 (avg.   0.0405777096) [mol/kg], -50.6530677727 (avg.   1.7853705296) [mg/g]
                                 -25.8038090266 (avg.   0.9095077991) [cm^3 STP/g],   -0.3342043241 (avg.   0.0117797120) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15822), density:   0.00000 (avg.   0.48771) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15822) [mol/uc],   0.0000000000 (avg.   1.1767910621) [mol/kg],   0.0000000000 (avg.  37.6559018388) [mg/g]
                                   0.0000000000 (avg.  26.3765663265) [cm^3 STP/g],    0.0000000000 (avg.   0.3416225299) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0034356196) [mol/uc],  -1.1512374831 (avg.   0.0255535790) [mol/kg], -36.8382179752 (avg.   0.8176838636) [mg/g]
                                 -25.8038090266 (avg.   0.5727572999) [cm^3 STP/g],   -0.3342043241 (avg.   0.0074182058) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15748), density:   0.00000 (avg.   0.42497) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15748) [mol/uc],   0.0000000000 (avg.   1.1712871727) [mol/kg],   0.0000000000 (avg.  32.8118297866) [mg/g]
                                   0.0000000000 (avg.  26.2532022823) [cm^3 STP/g],    0.0000000000 (avg.   0.3400247504) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0026956344) [mol/uc],  -1.1512374831 (avg.   0.0200496896) [mol/kg], -32.2501682089 (avg.   0.5616615776) [mg/g]
                                 -25.8038090266 (avg.   0.4493932557) [cm^3 STP/g],   -0.3342043241 (avg.   0.0058204263) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841137 [K]  (avg.          -5.2158417361)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.1993856473)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.0000000002 [K]  (avg.          -0.7676757102)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 60000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 60000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 2/0/0 (avg.   0.15953), density:   8.47708 (avg.   0.67618) [kg/m^3]
            absolute adsorption:   2.00000 (avg.   0.15953) [mol/uc],  14.8756743515 (avg.   1.1865631814) [mol/kg], 654.5118206577 (avg.  52.2073561052) [mg/g]
                                 333.4230041441 (avg.  26.5955983706) [cm^3 STP/g],    4.3184093337 (avg.   0.3444593769) [cm^3 STP/cm^3]
            excess adsorption:     1.8452187839 (avg.   0.0047494584) [mol/uc],  13.7244368684 (avg.   0.0353256983) [mol/kg], 603.8587528850 (avg.   1.5542883325) [mg/g]
                                 307.6191951175 (avg.   0.7917893440) [cm^3 STP/g],    3.9842050096 (avg.   0.0102550527) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15746), density:   0.00000 (avg.   0.48539) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15746) [mol/uc],   0.0000000000 (avg.   1.1711919074) [mol/kg],   0.0000000000 (avg.  37.4767356069) [mg/g]
                                   0.0000000000 (avg.  26.2510670085) [cm^3 STP/g],    0.0000000000 (avg.   0.3399970949) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0026828262) [mol/uc],  -1.1512374831 (avg.   0.0199544243) [mol/kg], -36.8382179752 (avg.   0.6385176317) [mg/g]
                                 -25.8038090266 (avg.   0.4472579819) [cm^3 STP/g],   -0.3342043241 (avg.   0.0057927708) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15656), density:   0.00000 (avg.   0.42251) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15656) [mol/uc],   0.0000000000 (avg.   1.1644979655) [mol/kg],   0.0000000000 (avg.  32.6216404671) [mg/g]
                                   0.0000000000 (avg.  26.1010291573) [cm^3 STP/g],    0.0000000000 (avg.   0.3380538431) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0017828412) [mol/uc],  -1.1512374831 (avg.   0.0132604824) [mol/kg], -32.2501682089 (avg.   0.3714722582) [mg/g]
                                 -25.8038090266 (avg.   0.2972201307) [cm^3 STP/g],   -0.3342043241 (avg.   0.0038495190) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 10 0 10 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           2 (2 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.5259949808 [K]  (avg.          -5.1842018640)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0948255242 [K]  (avg.          -4.1886078473)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.4856833275 [K]  (avg.          -0.7474684414)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 70000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 70000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15948), density:   0.00000 (avg.   0.67598) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15948) [mol/uc],   0.0000000000 (avg.   1.1862118288) [mol/kg],   0.0000000000 (avg.  52.1918970145) [mg/g]
                                   0.0000000000 (avg.  26.5877231630) [cm^3 STP/g],    0.0000000000 (avg.   0.3443573792) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0047022199) [mol/uc],  -1.1512374831 (avg.   0.0349743457) [mol/kg], -50.6530677727 (avg.   1.5388292418) [mg/g]
                                 -25.8038090266 (avg.   0.7839141364) [cm^3 STP/g],   -0.3342043241 (avg.   0.0101530551) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15695), density:   0.00000 (avg.   0.48382) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15695) [mol/uc],   0.0000000000 (avg.   1.1674049949) [mol/kg],   0.0000000000 (avg.  37.3555589518) [mg/g]
                                   0.0000000000 (avg.  26.1661872440) [cm^3 STP/g],    0.0000000000 (avg.   0.3388977540) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0021736846) [mol/uc],  -1.1512374831 (avg.   0.0161675118) [mol/kg], -36.8382179752 (avg.   0.5173409766) [mg/g]
                                 -25.8038090266 (avg.   0.3623782174) [cm^3 STP/g],   -0.3342043241 (avg.   0.0046934298) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15684), density:   0.00000 (avg.   0.42325) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15684) [mol/uc],   0.0000000000 (avg.   1.1665549685) [mol/kg],   0.0000000000 (avg.  32.6792642801) [mg/g]
                                   0.0000000000 (avg.  26.1471347731) [cm^3 STP/g],    0.0000000000 (avg.   0.3386509912) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0020594005) [mol/uc],  -1.1512374831 (avg.   0.0153174854) [mol/kg], -32.2501682089 (avg.   0.4290960711) [mg/g]
                                 -25.8038090266 (avg.   0.3433257465) [cm^3 STP/g],   -0.3342043241 (avg.   0.0044466671) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841134 [K]  (avg.          -5.0935563630)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.1436242019)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.0000000004 [K]  (avg.          -0.7019253025)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 80000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 80000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15971), density:   0.00000 (avg.   0.67694) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15971) [mol/uc],   0.0000000000 (avg.   1.1879007212) [mol/kg],   0.0000000000 (avg.  52.2662062508) [mg/g]
                                   0.0000000000 (avg.  26.6255779550) [cm^3 STP/g],    0.0000000000 (avg.   0.3448476648) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0049292875) [mol/uc],  -1.1512374831 (avg.   0.0366632380) [mol/kg], -50.6530677727 (avg.   1.6131384781) [mg/g]
                                 -25.8038090266 (avg.   0.8217689284) [cm^3 STP/g],   -0.3342043241 (avg.   0.0106433406) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15699), density:   0.00000 (avg.   0.48391) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15699) [mol/uc],   0.0000000000 (avg.   1.1676328682) [mol/kg],   0.0000000000 (avg.  37.3628506236) [mg/g]
                                   0.0000000000 (avg.  26.1712947903) [cm^3 STP/g],    0.0000000000 (avg.   0.3389639056) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0022043216) [mol/uc],  -1.1512374831 (avg.   0.0163953851) [mol/kg], -36.8382179752 (avg.   0.5246326484) [mg/g]
                                 -25.8038090266 (avg.   0.3674857638) [cm^3 STP/g],   -0.3342043241 (avg.   0.0047595815) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15706), density:   0.00000 (avg.   0.42385) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15706) [mol/uc],   0.0000000000 (avg.   1.1681906990) [mol/kg],   0.0000000000 (avg.  32.7250867836) [mg/g]
                                   0.0000000000 (avg.  26.1837979967) [cm^3 STP/g],    0.0000000000 (avg.   0.3391258439) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0022793207) [mol/uc],  -1.1512374831 (avg.   0.0169532159) [mol/kg], -32.2501682089 (avg.   0.4749185747) [mg/g]
                                 -25.8038090266 (avg.   0.3799889701) [cm^3 STP/g],   -0.3342043241 (avg.   0.0049215198) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841132 [K]  (avg.          -5.1415117268)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.1885759832)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.0000000006 [K]  (avg.          -0.7047357657)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Average Properties at Current cycle: 90000 out of 100000
        ========================================================================================

        Framework surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Framework 0 individual surface area:       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
            Cation surface area:                       0.0000000000 [m^2/g]        0.0000000000 [m^2/cm^3]       0.0000000000 [A^2]
        Compressibility:               -nan [-]
        Henry coefficients
            Component 0: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 1: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
            Component 2: 0 [mol/kg/Pa] (Rosenbluth factor new: 0 [-])
        Energy <U_gh>_1-<U_h>_0 from Widom


        Current cycle: 90000 out of 100000
        ========================================================================================================

        Net charge: 0 (F: 0, A: 0, C: 0)
        Current Box:  25.83200   0.00000   0.00000 [A]   Average Box:  25.83200   0.00000   0.00000 [A]
                       0.00000  25.83200   0.00000 [A]                  0.00000  25.83200   0.00000 [A]
                       0.00000   0.00000  25.83200 [A]                  0.00000   0.00000  25.83200 [A]
        Box-lengths:   25.83200  25.83200  25.83200 [A] Average:  25.83200  25.83200  25.83200 [A]
        Box-angles:   90.00000  90.00000  90.00000 [degrees] Average:  90.00000  90.00000  90.00000 [degrees]
        Volume: 17237.49273 [A^3] Average Volume: 17237.49273 [A^3]

        Loadings per component:
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Component 0 (CO2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15966), density:   0.00000 (avg.   0.67675) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15966) [mol/uc],   0.0000000000 (avg.   1.1875614739) [mol/kg],   0.0000000000 (avg.  52.2512797794) [mg/g]
                                   0.0000000000 (avg.  26.6179740756) [cm^3 STP/g],    0.0000000000 (avg.   0.3447491813) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0048836765) [mol/uc],  -1.1512374831 (avg.   0.0363239908) [mol/kg], -50.6530677727 (avg.   1.5982120067) [mg/g]
                                 -25.8038090266 (avg.   0.8141650490) [cm^3 STP/g],   -0.3342043241 (avg.   0.0105448571) [cm^3 STP/cm^3]
        Component 1 (O2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15656), density:   0.00000 (avg.   0.48262) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15656) [mol/uc],   0.0000000000 (avg.   1.1645044349) [mol/kg],   0.0000000000 (avg.  37.2627445108) [mg/g]
                                   0.0000000000 (avg.  26.1011741614) [cm^3 STP/g],    0.0000000000 (avg.   0.3380557212) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0017837110) [mol/uc],  -1.1512374831 (avg.   0.0132669518) [mol/kg], -36.8382179752 (avg.   0.4245265357) [mg/g]
                                 -25.8038090266 (avg.   0.2973651348) [cm^3 STP/g],   -0.3342043241 (avg.   0.0038513970) [cm^3 STP/cm^3]
        Component 2 (N2), current number of integer/fractional/reaction molecules: 0/0/0 (avg.   0.15745), density:   0.00000 (avg.   0.42491) [kg/m^3]
            absolute adsorption:   0.00000 (avg.   0.15745) [mol/uc],   0.0000000000 (avg.   1.1711157722) [mol/kg],   0.0000000000 (avg.  32.8070282634) [mg/g]
                                   0.0000000000 (avg.  26.2493605167) [cm^3 STP/g],    0.0000000000 (avg.   0.3399749929) [cm^3 STP/cm^3]
            excess adsorption:    -0.1547812161 (avg.   0.0026725900) [mol/uc],  -1.1512374831 (avg.   0.0198782891) [mol/kg], -32.2501682089 (avg.   0.5568600545) [mg/g]
                                 -25.8038090266 (avg.   0.4455514901) [cm^3 STP/g],   -0.3342043241 (avg.   0.0057706688) [cm^3 STP/cm^3]
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        Degrees of freedom: 0 0 0 0
        Number of Framework-atoms:      7
        Number of Adsorbates:           0 (0 integer, 0 fractional, 0 reaction)
        Number of Cations:              0 (0 integer, 0 fractional, 0 reaction)

        Current total potential energy:                -0.1295841131 [K]  (avg.          -5.0711425880)
            Current Host-Host energy:                     0.0000000000 [K]  (avg.           0.0000000000)
            Current Host-Adsorbate energy:               -0.0000000000 [K]  (avg.          -4.1154704449)
            Current Host-Cation energy:                   0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Adsorbate energy:           0.0000000007 [K]  (avg.          -0.7075112058)
            Current Cation-Cation energy:                 0.0000000000 [K]  (avg.           0.0000000000)
            Current Adsorbate-Cation energy:              0.0000000000 [K]  (avg.           0.0000000000)

        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Finishing simulation
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




        Current (running energy) Energy Status
        ===========================================================================

        Internal energy:
        Host stretch energy:                                            0.00000000
        Host UreyBradley energy:                                        0.00000000
        Host bend energy:                                               0.00000000
        Host inversion-bend energy:                                     0.00000000
        Host torsion energy:                                            0.00000000
        Host improper torsion energy:                                   0.00000000
        Host out-of-plane energy:                                       0.00000000
        Host stretch/stretch energy:                                    0.00000000
        Host bend/bend energy:                                          0.00000000
        Host stretch/bend energy:                                       0.00000000
        Host stretch/torsion energy:                                    0.00000000
        Host bend/torsion energy:                                       0.00000000

        Adsorbate stretch energy:                                       0.00000000
        Adsorbate UreyBradley energy:                                   0.00000000
        Adsorbate bend energy:                                          0.00000000
        Adsorbate inversion-bend energy:                                0.00000000
        Adsorbate torsion energy:                                       0.00000000
        Adsorbate improper torsion energy:                              0.00000000
        Adsorbate out-of-plane energy:                                  0.00000000
        Adsorbate stretch/stretch energy:                               0.00000000
        Adsorbate bend/bend energy:                                     0.00000000
        Adsorbate stretch/bend energy:                                  0.00000000
        Adsorbate stretch/torsion energy:                               0.00000000
        Adsorbate bend/torsion energy:                                  0.00000000
        Adsorbate intra VDW energy:                                     0.00000000
        Adsorbate intra charge-charge Coulomb energy:                   0.00000000
        Adsorbate intra charge-bonddipole Coulomb energy:               0.00000000
        Adsorbate intra bonddipole-bonddipole Coulomb energy:           0.00000000

        Cation stretch energy:                                          0.00000000
        Cation UreyBradley energy:                                      0.00000000
        Cation bend energy:                                             0.00000000
        Cation inversion-bend energy:                                   0.00000000
        Cation torsion energy:                                          0.00000000
        Cation improper torsion energy:                                 0.00000000
        Cation out-of-plane energy:                                     0.00000000
        Cation stretch/stretch energy:                                  0.00000000
        Cation bend/bend energy:                                        0.00000000
        Cation stretch/bend energy:                                     0.00000000
        Cation stretch/torsion energy:                                  0.00000000
        Cation bend/torsion energy:                                     0.00000000
        Cation intra VDW energy:                                        0.00000000
        Cation intra charge-charge Coulomb energy:                      0.00000000
        Cation intra charge-bonddipole Coulomb energy:                  0.00000000
        Cation intra bonddipole-bonddipole Coulomb energy:              0.00000000

        Host/Host energy:                                             0.00000000
            Host/Host VDW energy:                                         0.00000000
            Host/Host Coulomb energy:                                     0.00000000
            Host/Host charge-charge Real energy:                          0.00000000
            Host/Host charge-charge Fourier energy:                       0.00000000
            Host/Host charge-bonddipole Real energy:                      0.00000000
            Host/Host charge-bonddipole Fourier energy:                   0.00000000
            Host/Host bondipole-bonddipole Real energy:                   0.00000000
            Host/Host bondipole-bonddipole Fourier energy:                0.00000000

        Host/Adsorbate energy:                                      -25.55362524
            Host/Adsorbate VDW energy:                                  -25.55362524
            Host/Adsorbate Coulomb energy:                                0.00000000
            Host/Adsorbate charge-charge Real energy:                     0.00000000
            Host/Adsorbate charge-charge Fourier energy:                  0.00000000
            Host/Adsorbate charge-bonddipole Real energy:                 0.00000000
            Host/Adsorbate charge-bonddipole Fourier energy:              0.00000000
            Host/Adsorbate bondipole-bonddipole Real energy:              0.00000000
            Host/Adsorbate bondipole-bonddipole Fourier energy:           0.00000000

        Host/Cation energy:                                           0.00000000
            Host/Cation VDW energy:                                       0.00000000
            Host/Cation Coulomb energy:                                   0.00000000
            Host/Cation charge-charge Real energy:                        0.00000000
            Host/Cation charge-charge Fourier energy:                     0.00000000
            Host/Cation charge-bonddipole Real energy:                    0.00000000
            Host/Cation charge-bonddipole Fourier energy:                 0.00000000
            Host/Cation bondipole-bonddipole Real energy:                 0.00000000
            Host/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Adsorbate/Adsorbate energy:                                       -0.00017471
            Adsorbate/Adsorbate VDW energy:                                    0.00000000
            Adsorbate/Adsorbate Coulomb energy:                               -0.00017471
            Adsorbate/Adsorbate charge-charge Real energy:                     0.00000000
            Adsorbate/Adsorbate charge-charge Fourier energy:                 -0.00017471
            Adsorbate/Adsorbate charge-bonddipole Real energy:                 0.00000000
            Adsorbate/Adsorbate charge-bonddipole Fourier energy:              0.00000000
            Adsorbate/Adsorbate bondipole-bonddipole Real energy:              0.00000000
            Adsorbate/Adsorbate bondipole-bonddipole Fourier energy:           0.00000000

        Adsorbate/Cation energy:                                           0.00000000
            Adsorbate/Cation VDW energy:                                       0.00000000
            Adsorbate/Cation Coulomb energy:                                   0.00000000
            Adsorbate/Cation charge-charge Real energy:                        0.00000000
            Adsorbate/Cation charge-charge Fourier energy:                     0.00000000
            Adsorbate/Cation charge-bonddipole Real energy:                    0.00000000
            Adsorbate/Cation charge-bonddipole Fourier energy:                 0.00000000
            Adsorbate/Cation bondipole-bonddipole Real energy:                 0.00000000
            Adsorbate/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Cation/Cation energy:                                           0.00000000
            Cation/Cation VDW energy:                                       0.00000000
            Cation/Cation Coulomb energy:                                   0.00000000
            Cation/Cation charge-charge Real energy:                        0.00000000
            Cation/Cation charge-charge Fourier energy:                     0.00000000
            Cation/Cation charge-bonddipole Real energy:                    0.00000000
            Cation/Cation charge-bonddipole Fourier energy:                 0.00000000
            Cation/Cation bondipole-bonddipole Real energy:                 0.00000000
            Cation/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Polarization energy:
            Host polarization energy:                                     0.00000000
            Adsorbate polarization energy:                                0.00000000
            Cation polarization energy:                                   0.00000000
            Host back-polarization energy:                                     0.00000000
            Adsorbate back-polarization energy:                                0.00000000
            Cation back-polarization energy:                                   0.00000000

        Tail-correction energy:                                      -0.29878736

        Distance constraints energy:                                  0.00000000
        Angle constraints energy:                                     0.00000000
        Dihedral constraints energy:                                  0.00000000
        Inversion-bend constraints energy:                            0.00000000
        Out-of-plane distance constraints energy:                     0.00000000
        Exclusion constraints energy:                                 0.00000000

        ===================================================================
        Total energy:   -25.852587306066
            Total Van der Waals: -25.553625
            Total Coulomb: -0.000175

            Total Polarization: 0.000000

        Monte-Carlo moves statistics
        ===========================================================================

        Performance of the small-MC scheme
        ==================================

        Component 0 [CO2]
        ----------------------------------------------
        Bead: 0
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]

        Bead: 1
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]
            maximum change bend angle           : 0.300000
            change bend angle acceptence        : 0.000000 [%]

        Bead: 2
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]

        Component 1 [O2]
        ----------------------------------------------
        Bead: 0
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]

        Bead: 1
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]
            maximum change bend angle           : 0.300000
            change bend angle acceptence        : 0.000000 [%]

        Bead: 2
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]

        Component 2 [N2]
        ----------------------------------------------
        Bead: 0
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]

        Bead: 1
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]
            maximum change bend angle           : 0.300000
            change bend angle acceptence        : 0.000000 [%]

        Bead: 2
            maximum bond length change          : 0.300000
            bond length change acceptence       : 0.000000 [%]



        Performance of the translation move:
        ======================================
        Component 0 [CO2]
            total        19442.000000 19757.000000 19559.000000
            succesfull   19204.000000 19520.000000 19407.000000
            accepted   0.987758 0.988004 0.992229
            displacement 1.000000 1.000000 1.000000

        Component 1 [O2]
            total        19061.000000 19421.000000 19227.000000
            succesfull   18919.000000 19271.000000 19122.000000
            accepted   0.992550 0.992276 0.994539
            displacement 1.000000 1.000000 1.000000

        Component 2 [N2]
            total        19424.000000 19227.000000 19444.000000
            succesfull   19311.000000 19069.000000 19365.000000
            accepted   0.994182 0.991782 0.995937
            displacement 1.000000 1.000000 1.000000


        Random translation move was OFF for all components

        Performance of the rotation move:
        =================================
        Component 0 [CO2]
            total        19444.000000 19543.000000 19463.000000
            succesfull   19145.000000 19208.000000 19098.000000
            accepted   0.984623 0.982858 0.981246
            angle-change 180.000000 180.000000 180.000000

        Component 1 [O2]
            total        19336.000000 19264.000000 19269.000000
            succesfull   19196.000000 19134.000000 19152.000000
            accepted   0.992760 0.993252 0.993928
            angle-change 180.000000 180.000000 180.000000

        Component 2 [N2]
            total        19400.000000 19339.000000 19490.000000
            succesfull   19291.000000 19234.000000 19362.000000
            accepted   0.994381 0.994571 0.993433
            angle-change 180.000000 180.000000 180.000000


        Random rotation move was OFF for all components

        Performance of the swap addition move:
        ======================================
        Component [CO2] total tried: 200020.000000 succesfull growth: 200020.000000 (100.000000 [%]) accepted: 29077.000000 (14.537046 [%])
        Component [O2] total tried: 200243.000000 succesfull growth: 200243.000000 (100.000000 [%]) accepted: 29019.000000 (14.491892 [%])
        Component [N2] total tried: 200402.000000 succesfull growth: 200402.000000 (100.000000 [%]) accepted: 28880.000000 (14.411034 [%])

        Performance of the swap deletion move:
        ======================================
        Component [CO2] total tried: 199303.000000 succesfull growth: 29088.000000 (14.594863 [%]) accepted: 29088.000000 (14.594863 [%])
        Component [O2] total tried: 200705.000000 succesfull growth: 28860.000000 (14.379313 [%]) accepted: 28860.000000 (14.379313 [%])
        Component [N2] total tried: 199647.000000 succesfull growth: 29027.000000 (14.539162 [%]) accepted: 29027.000000 (14.539162 [%])

        Performance of the Reinsertion move:
        ====================================
        Component [CO2] total tried: 58189.000000 succesfull growth: 58189.000000 (100.000000 [%]) accepted: 56941.000000 (97.855265 [%])
        Component [O2] total tried: 57843.000000 succesfull growth: 57843.000000 (100.000000 [%]) accepted: 57055.000000 (98.637692 [%])
        Component [N2] total tried: 58194.000000 succesfull growth: 58194.000000 (100.000000 [%]) accepted: 57372.000000 (98.587483 [%])

        Reinsertion-in-plane move was OFF for all components

        Reinsertion-in-place move was OFF for all components

        Partial reinsertion move was OFF for all components

        Performance of the identity change move:
        ======================================
        Component [CO2]->[CO2] total tried: 19540.000000 succesfull growth: 19540.000000 (100.000000 [%]) accepted: 10010.000000 (51.228250 [%])
        Component [CO2]->[O2] total tried: 19382.000000 succesfull growth: 19382.000000 (100.000000 [%]) accepted: 17849.000000 (92.090600 [%])
        Component [CO2]->[N2] total tried: 19813.000000 succesfull growth: 19813.000000 (100.000000 [%]) accepted: 18142.000000 (91.566143 [%])
        Component [O2]->[CO2] total tried: 19471.000000 succesfull growth: 19471.000000 (100.000000 [%]) accepted: 18025.000000 (92.573571 [%])
        Component [O2]->[O2] total tried: 19250.000000 succesfull growth: 19250.000000 (100.000000 [%]) accepted: 9902.000000 (51.438961 [%])
        Component [O2]->[N2] total tried: 19230.000000 succesfull growth: 19230.000000 (100.000000 [%]) accepted: 17859.000000 (92.870515 [%])
        Component [N2]->[CO2] total tried: 19439.000000 succesfull growth: 19439.000000 (100.000000 [%]) accepted: 17977.000000 (92.479037 [%])
        Component [N2]->[O2] total tried: 19227.000000 succesfull growth: 19227.000000 (100.000000 [%]) accepted: 17877.000000 (92.978624 [%])
        Component [N2]->[N2] total tried: 19336.000000 succesfull growth: 19336.000000 (100.000000 [%]) accepted: 9904.000000 (51.220521 [%])

        Parallel tempering move was OFF

        Hyper parallel tempering move was OFF

        Parallel mol-fraction move was OFF

        Chiral inversion move was OFF

        Volume move was OFF

        Box shape change move was OFF

        Framework change move was OFF

        Framework shift move was OFF

        Hybrid MC/MD move in the NVE-ensemble was OFF

        Hybrid MC/MD in the NPH-ensemble move was OFF

        Hybrid MC/MD in the NPH-ensemble (Parrinello-Rahman) move was OFF

        Gibbs volume change move was OFF

        Gibbs swap move was OFF for all components

        Gibbs identity change move was OFF for all components

        CFCMC swap lambda move was OFF for all components

        CB/CFCMC swap lambda move was OFF for all components

        CFCMC Gibbs lambda move was OFF for all components

        CB/CFCMC Gibbs lambda move was OFF for all components

        No reactions present, RXMC is OFF

        Exchange fractional-particle move was OFF for all components

        CFCMC Gibbs Lambda-change move was OFF for all components

        CFCMC Gibbs Swap-Fractional-Molecule-To-Other-Box move was OFF for all components

        CFCMC Gibbs Swap-Fractional-Molecule-To-Other-Box move was OFF for all components

        CFCMC swap lambda move was OFF for all components

        Gibbs Widom move was OFF for all components



        Total CPU timings:
        ===========================================
        initialization:              7.483968 [s]
        equilibration:                      0 [s]
        production run:             78.253071 [s]
        total time:                 85.737039 [s]

        Production run CPU timings of the MC moves:
        ===========================================
        Component: 0 (CO2)
            translation:                                  1.974067 [s]
            random translation:                                  0 [s]
            rotation:                                      1.96932 [s]
            random rotation:                                     0 [s]
            partial reinsertion:                                 0 [s]
            reinsertion:                                  3.926411 [s]
            reinsertion in-place:                                0 [s]
            reinsertion in-plane:                                0 [s]
            identity switch:                              5.013137 [s]
            swap (insertion):                            11.295829 [s]
            swap (deletion):                              1.670342 [s]
            swap lambda (CFMC):                                  0 [s]
            swap lambda (CB/CFMC):                               0 [s]
            Widom:                                               0 [s]
            CF-Widom:                                            0 [s]
            Gibbs Widom:                                         0 [s]
            surface area:                                        0 [s]
            Gibbs particle transform:                            0 [s]
            Gibbs particle transform (CFMC):                     0 [s]
            Gibbs particle transform (CB/CFMC):                  0 [s]
            Gibbs indentity change:                              0 [s]
            Exchange fract./int. particle:                       0 [s]
            Swap Gibbs-fractional molecules:                     0 [s]
            Change Gibs-lambda value:                            0 [s]
            Convert Gibbs fract. to integer:                     0 [s]
        Component: 1 (O2)
            translation:                                  1.932714 [s]
            random translation:                                  0 [s]
            rotation:                                     1.945144 [s]
            random rotation:                                     0 [s]
            partial reinsertion:                                 0 [s]
            reinsertion:                                  3.832731 [s]
            reinsertion in-place:                                0 [s]
            reinsertion in-plane:                                0 [s]
            identity switch:                              4.929783 [s]
            swap (insertion):                            11.195834 [s]
            swap (deletion):                               1.63997 [s]
            swap lambda (CFMC):                                  0 [s]
            swap lambda (CB/CFMC):                               0 [s]
            Widom:                                               0 [s]
            CF-Widom:                                            0 [s]
            Gibbs Widom:                                         0 [s]
            surface area:                                        0 [s]
            Gibbs particle transform:                            0 [s]
            Gibbs particle transform (CFMC):                     0 [s]
            Gibbs particle transform (CB/CFMC):                  0 [s]
            Gibbs indentity change:                              0 [s]
            Exchange fract./int. particle:                       0 [s]
            Swap Gibbs-fractional molecules:                     0 [s]
            Change Gibs-lambda value:                            0 [s]
            Convert Gibbs fract. to integer:                     0 [s]
        Component: 2 (N2)
            translation:                                  1.944647 [s]
            random translation:                                  0 [s]
            rotation:                                     1.954243 [s]
            random rotation:                                     0 [s]
            partial reinsertion:                                 0 [s]
            reinsertion:                                  3.851865 [s]
            reinsertion in-place:                                0 [s]
            reinsertion in-plane:                                0 [s]
            identity switch:                              4.950372 [s]
            swap (insertion):                            11.197773 [s]
            swap (deletion):                              1.647581 [s]
            swap lambda (CFMC):                                  0 [s]
            swap lambda (CB/CFMC):                               0 [s]
            Widom:                                               0 [s]
            CF-Widom:                                            0 [s]
            Gibbs Widom:                                         0 [s]
            surface area:                                        0 [s]
            Gibbs particle transform:                            0 [s]
            Gibbs particle transform (CFMC):                     0 [s]
            Gibbs particle transform (CB/CFMC):                  0 [s]
            Gibbs indentity change:                              0 [s]
            Exchange fract./int. particle:                       0 [s]
            Swap Gibbs-fractional molecules:                     0 [s]
            Change Gibs-lambda value:                            0 [s]
            Convert Gibbs fract. to integer:                     0 [s]

        Total all components:
            translation:                                  5.851428 [s]
            random translation:                                  0 [s]
            rotation:                                     5.868707 [s]
            random rotation:                                     0 [s]
            partial reinsertion:                                 0 [s]
            reinsertion:                                 11.611007 [s]
            reinsertion in-place:                                0 [s]
            reinsertion in-plane:                                0 [s]
            identity switch:                             14.893292 [s]
            swap (insertion):                            33.689436 [s]
            swap (deletion):                             33.689436 [s]
            swap lambda (CFMC):                                  0 [s]
            swap lambda (CB/CFMC):                               0 [s]
            Widom:                                               0 [s]
            CF-Widom:                                            0 [s]
            Gibbs Widom:                                         0 [s]
            surface area:                                        0 [s]
            Gibbs particle transform:                            0 [s]
            Gibbs particle transform (CFMC):                     0 [s]
            Gibbs particle transform (CB/CFMC):                  0 [s]
            Gibbs identity change:                               0 [s]
            Exchange fract./int. particle:                       0 [s]
            Swap Gibbs-fractional molecules:                     0 [s]
            Change Gibs-lambda value:                            0 [s]
            Convert Gibbs fract. to integer:                     0 [s]

        System moves:
            parallel tempering:                             0 [s]
            hyper parallel tempering:                       0 [s]
            mol-fraction replica-exchange:                  0 [s]
            chiral inversion:                               0 [s]
            hybrid MC/MD (NVE):                             0 [s]
            hybrid MC/MD (NPH):                             0 [s]
            hybrid MC/MD (NPHPR):                           0 [s]
            volume change:                                  0 [s]
            box change:                                     0 [s]
            Gibbs volume change:                            0 [s]
            framework change:                               0 [s]
            framework shift:                                0 [s]
            reaction MC move:                               0 [s]

        Production run CPU timings of the MC moves summed over all systems and components:
        ==================================================================================

        Particles moves:
            translation:                                  5.851428 [s]
            random translation:                                  0 [s]
            rotation:                                     5.868707 [s]
            random rotation:                                     0 [s]
            partial reinsertion:                                 0 [s]
            reinsertion:                                 11.611007 [s]
            reinsertion in-place:                                0 [s]
            reinsertion in-plane:                                0 [s]
            identity switch:                             14.893292 [s]
            swap (insertion):                            33.689436 [s]
            swap (deletion):                             33.689436 [s]
            swap lambda (CFMC):                                  0 [s]
            swap lambda (CB/CFMC):                               0 [s]
            Widom:                                               0 [s]
            CF-Widom:                                            0 [s]
            Gibbs Widom:                                         0 [s]
            surface area:                                        0 [s]
            Gibbs particle transform:                            0 [s]
            Gibbs particle transform (CFMC):                     0 [s]
            Gibbs particle transform (CB/CFMC):                  0 [s]
            Gibbs indentity change:                              0 [s]
            Exchange frac./int. particle:                        0 [s]
            Swap Gibbs-fractional molecules:                     0 [s]
            Change Gibs-lambda value:                            0 [s]
            Convert Gibbs fract. to integer:                     0 [s]

        System moves:
            parallel tempering:                             0 [s]
            hyper parallel tempering:                       0 [s]
            mol-fraction replica-exchange:                  0 [s]
            chiral inversion:                               0 [s]
            hybrid MC/MD (NVE):                             0 [s]
            hybrid MC/MD (NPH):                             0 [s]
            hybrid MC/MD (NPHPR):                           0 [s]
            volume change:                                  0 [s]
            box change:                                     0 [s]
            Gibbs volume change:                            0 [s]
            framework change:                               0 [s]
            framework shift:                                0 [s]
            reaction MC move:                               0 [s]





        Current (full final energy) Energy Status
        ===========================================================================

        Internal energy:
        Host stretch energy:                                            0.00000000
        Host UreyBradley energy:                                        0.00000000
        Host bend energy:                                               0.00000000
        Host inversion-bend energy:                                     0.00000000
        Host torsion energy:                                            0.00000000
        Host improper torsion energy:                                   0.00000000
        Host out-of-plane energy:                                       0.00000000
        Host stretch/stretch energy:                                    0.00000000
        Host bend/bend energy:                                          0.00000000
        Host stretch/bend energy:                                       0.00000000
        Host stretch/torsion energy:                                    0.00000000
        Host bend/torsion energy:                                       0.00000000

        Adsorbate stretch energy:                                       0.00000000
        Adsorbate UreyBradley energy:                                   0.00000000
        Adsorbate bend energy:                                          0.00000000
        Adsorbate inversion-bend energy:                                0.00000000
        Adsorbate torsion energy:                                       0.00000000
        Adsorbate improper torsion energy:                              0.00000000
        Adsorbate out-of-plane energy:                                  0.00000000
        Adsorbate stretch/stretch energy:                               0.00000000
        Adsorbate bend/bend energy:                                     0.00000000
        Adsorbate stretch/bend energy:                                  0.00000000
        Adsorbate stretch/torsion energy:                               0.00000000
        Adsorbate bend/torsion energy:                                  0.00000000
        Adsorbate intra VDW energy:                                     0.00000000
        Adsorbate intra charge-charge Coulomb energy:                   0.00000000
        Adsorbate intra charge-bonddipole Coulomb energy:               0.00000000
        Adsorbate intra bonddipole-bonddipole Coulomb energy:           0.00000000

        Cation stretch energy:                                          0.00000000
        Cation UreyBradley energy:                                      0.00000000
        Cation bend energy:                                             0.00000000
        Cation inversion-bend energy:                                   0.00000000
        Cation torsion energy:                                          0.00000000
        Cation improper torsion energy:                                 0.00000000
        Cation out-of-plane energy:                                     0.00000000
        Cation stretch/stretch energy:                                  0.00000000
        Cation bend/bend energy:                                        0.00000000
        Cation stretch/bend energy:                                     0.00000000
        Cation stretch/torsion energy:                                  0.00000000
        Cation bend/torsion energy:                                     0.00000000
        Cation intra VDW energy:                                        0.00000000
        Cation intra charge-charge Coulomb energy:                      0.00000000
        Cation intra charge-bonddipole Coulomb energy:                  0.00000000
        Cation intra bonddipole-bonddipole Coulomb energy:              0.00000000

        Host/Host energy:                                             0.00000000
            Host/Host VDW energy:                                         0.00000000
            Host/Host Coulomb energy:                                     0.00000000
            Host/Host charge-charge Real energy:                          0.00000000
            Host/Host charge-charge Fourier energy:                       0.00000000
            Host/Host charge-bonddipole Real energy:                      0.00000000
            Host/Host charge-bonddipole Fourier energy:                   0.00000000
            Host/Host bondipole-bonddipole Real energy:                   0.00000000
            Host/Host bondipole-bonddipole Fourier energy:                0.00000000

        Host/Adsorbate energy:                                      -25.55362524
            Host/Adsorbate VDW energy:                                  -25.55362524
            Host/Adsorbate Coulomb energy:                                0.00000000
            Host/Adsorbate charge-charge Real energy:                     0.00000000
            Host/Adsorbate charge-charge Fourier energy:                  0.00000000
            Host/Adsorbate charge-bonddipole Real energy:                 0.00000000
            Host/Adsorbate charge-bonddipole Fourier energy:              0.00000000
            Host/Adsorbate bondipole-bonddipole Real energy:              0.00000000
            Host/Adsorbate bondipole-bonddipole Fourier energy:           0.00000000

        Host/Cation energy:                                           0.00000000
            Host/Cation VDW energy:                                       0.00000000
            Host/Cation Coulomb energy:                                   0.00000000
            Host/Cation charge-charge Real energy:                        0.00000000
            Host/Cation charge-charge Fourier energy:                     0.00000000
            Host/Cation charge-bonddipole Real energy:                    0.00000000
            Host/Cation charge-bonddipole Fourier energy:                 0.00000000
            Host/Cation bondipole-bonddipole Real energy:                 0.00000000
            Host/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Adsorbate/Adsorbate energy:                                       -0.00017471
            Adsorbate/Adsorbate VDW energy:                                    0.00000000
            Adsorbate/Adsorbate Coulomb energy:                               -0.00017471
            Adsorbate/Adsorbate charge-charge Real energy:                     0.00000000
            Adsorbate/Adsorbate charge-charge Fourier energy:                 -0.00017471
            Adsorbate/Adsorbate charge-bonddipole Real energy:                 0.00000000
            Adsorbate/Adsorbate charge-bonddipole Fourier energy:              0.00000000
            Adsorbate/Adsorbate bondipole-bonddipole Real energy:              0.00000000
            Adsorbate/Adsorbate bondipole-bonddipole Fourier energy:           0.00000000

        Adsorbate/Cation energy:                                           0.00000000
            Adsorbate/Cation VDW energy:                                       0.00000000
            Adsorbate/Cation Coulomb energy:                                   0.00000000
            Adsorbate/Cation charge-charge Real energy:                        0.00000000
            Adsorbate/Cation charge-charge Fourier energy:                     0.00000000
            Adsorbate/Cation charge-bonddipole Real energy:                    0.00000000
            Adsorbate/Cation charge-bonddipole Fourier energy:                 0.00000000
            Adsorbate/Cation bondipole-bonddipole Real energy:                 0.00000000
            Adsorbate/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Cation/Cation energy:                                           0.00000000
            Cation/Cation VDW energy:                                       0.00000000
            Cation/Cation Coulomb energy:                                   0.00000000
            Cation/Cation charge-charge Real energy:                        0.00000000
            Cation/Cation charge-charge Fourier energy:                     0.00000000
            Cation/Cation charge-bonddipole Real energy:                    0.00000000
            Cation/Cation charge-bonddipole Fourier energy:                 0.00000000
            Cation/Cation bondipole-bonddipole Real energy:                 0.00000000
            Cation/Cation bondipole-bonddipole Fourier energy:              0.00000000

        Polarization energy:
            Host polarization energy:                                     0.00000000
            Adsorbate polarization energy:                                0.00000000
            Cation polarization energy:                                   0.00000000
            Host back-polarization energy:                                     0.00000000
            Adsorbate back-polarization energy:                                0.00000000
            Cation back-polarization energy:                                   0.00000000

        Tail-correction energy:                                      -0.29878736

        Distance constraints energy:                                  0.00000000
        Angle constraints energy:                                     0.00000000
        Dihedral constraints energy:                                  0.00000000
        Inversion-bend constraints energy:                            0.00000000
        Out-of-plane distance constraints energy:                     0.00000000
        Exclusion constraints energy:                                 0.00000000

        ===================================================================
        Total energy:   -25.852587306777
            Total Van der Waals: -25.553625
            Total Coulomb: -0.000175

            Total Polarization: 0.000000



        Energy-drift status
        ===========================================================================

        Internal energy:
        Host stretch energy-drift:                                           0
        Host UreyBradley energy-drift:                                       0
        Host bend energy-drift:                                              0
        Host inversion-bend energy-drift:                                    0
        Host torsion energy-drift:                                           0
        Host torsion improper energy-drift:                                  0
        Host out-of-plane energy-drift:                                      0
        Host stretch/stretch energy-drift:                                   0
        Host stretch/bend energy-drift:                                      0
        Host bend/bend energy-drift:                                         0
        Host stretch/torsion energy-drift:                                   0
        Host bend/torsion energy-drift:                                      0

        Adsorbate stretch energy-drift:                                      0
        Adsorbate UreyBradley energy-drift:                                  0
        Adsorbate bend energy-drift:                                         0
        Adsorbate inversion-bend energy-drift:                               0
        Adsorbate torsion energy-drift:                                      0
        Adsorbate improper torsion energy-drift:                             0
        Adsorbate out-of-plane energy-drift:                                 0
        Adsorbate stretch/stretch energy-drift:                              0
        Adsorbate stretch/bend energy-drift:                                 0
        Adsorbate bend/bend energy-drift:                                    0
        Adsorbate stretch/torsion energy-drift:                              0
        Adsorbate bend/torsion energy-drift:                                 0
        Adsorbate intra VDW energy-drift:                                    0
        Adsorbate intra charge-charge Coulomb energy-drift:                  0
        Adsorbate intra charge-bonddipole Coulomb energy-drift:              0
        Adsorbate intra bonddipole-bonddipole Coulomb energy-drift:          0

        Cation stretch energy-drift:                                         0
        Cation UreyBradley energy-drift:                                     0
        Cation bend energy-drift:                                            0
        Cation inversion-bend energy-drift:                                  0
        Cation torsion energy-drift:                                         0
        Cation improper torsion energy-drift:                                0
        Cation out-of-plane energy-drift:                                    0
        Cation stretch/stretch energy-drift:                                 0
        Cation stretch/bend energy-drift:                                    0
        Cation bend/bend energy-drift:                                       0
        Cation stretch/torsion energy-drift:                                 0
        Cation bend/torsion energy-drift:                                    0
        Cation intra VDW energy-drift:                                       0
        Cation intra Coulomb charge-charge energy-drift:                     0
        Cation intra Coulomb charge-bonddipole energy-drift:                 0
        Cation intra Coulomb bonddipole-bonddipole energy-drift:             0

        Host/Host energy-drift:                                              0
            Host/Host VDW energy-drift:                                        0
            Host/Host Coulomb energy-drift:                                    0
                Host/Host Real charge-charge energy-drift:                       0
                Host/Host Fourier charge-charge energy-drift:                    0
                Host/Host Real charge-bonddipole energy-drift:                   0
                Host/Host Fourier charge-bonddipole energy-drift:                0
                Host/Host Real bonddipole-bonddipole energy-drift:               0
                Host/Host Fourier bonddipole-bonddipole energy-drift:            0
        Host/Adsorbate energy-drift:                                         -1.97837e-12
            Host/Adsorbate VDW energy-drift:                                   -1.97837e-12
            Host/Adsorbate Coulomb energy-drift:                               0
                Host/Adsorbate Real charge-charge energy-drift:                  0
                Host/Adsorbate Fourier charge-charge energy-drift:               0
                Host/Adsorbate Real charge-bonddipole energy-drift:              0
                Host/Adsorbate Fourier charge-bonddipole energy-drift:           0
                Host/Adsorbate Real bonddipole-bonddipole energy-drift:          0
                Host/Adsorbate Fourier bonddipole-bonddipole energy-drift:       0
        Host/Cation energy-drift:                                            0
            Host/Cation VDW energy-drift:                                      0
            Host/Cation Coulomb energy-drift:                                  0
                Host/Cation Real charge-charge energy-drift:                     0
                Host/Cation Fourier charge-charge energy-drift:                  0
                Host/Cation Real charge-bonddipole energy-drift:                 0
                Host/Cation Fourier charge-bonddipole energy-drift:              0
                Host/Cation Real bonddipole-bonddipole energy-drift:             0
                Host/Cation Fourier bonddipole-bonddipole energy-drift:          0
        Adsorbate/Adsorbate energy-drift:                                     7.17217e-10
            Adsorbate/Adsorbate VDW energy-drift:                               7.35779e-13
            Adsorbate/Adsorbate Coulomb energy-drift:                           7.1608e-10
                Adsorbate/Adsorbate Real charge-charge energy-drift:              1.25127e-11
                Adsorbate/Adsorbate Fourier charge-charge energy-drift:           7.03603e-10
                Adsorbate/Adsorbate Real charge-bonddipole energy-drift:          0
                Adsorbate/Adsorbate Fourier charge-bonddipole energy-drift:       0
                Adsorbate/Adsorbate Real bonddipole-bonddipole energy-drift:      0
                Adsorbate/Adsorbate Fourier bonddipole-bonddipole energy-drift:   0
        Cation/Cation energy-drift:                                           0
            Cation/Cation VDW energy-drift:                                     0
            Cation/Cation Coulomb energy-drift:                                 0
                Cation/Cation Real charge-charge energy-drift:                    0
                Cation/Cation Fourier charge-charge energy-drift:                 0
                Cation/Cation Real charge-bonddipole energy-drift:                0
                Cation/Cation Fourier charge-bonddipole energy-drift:             0
                Cation/Cation Real bonddipole-bonddipole energy-drift:            0
                Cation/Cation Fourier bonddipole-bonddipole energy-drift:         0
        Adsorbate/Cation energy-drift:                                        0
            Adsorbate/Cation VDW energy-drift:                                  0
            Adsorbate/Cation Coulomb energy-drift:                              0
                Adsorbate/Cation Real charge-charge energy-drift:                 0
                Adsorbate/Cation Fourier charge-charge energy-drift:              0
                Adsorbate/Cation Real charge-bonddipole energy-drift:             0
                Adsorbate/Cation Fourier charge-bonddipole energy-drift:          0
                Adsorbate/Cation Real bonddipole-bonddipole energy-drift:         0
                Adsorbate/Cation Fourier bonddipole-bonddipole energy-drift:      0

        Polarization energy-drift:
            Host polarization energy-drift:                0
            Adsorbate polarization energy-drift:           0
            Cation polarization energy-drift:              0
            Host back-polarization energy-drift:                0
            Adsorbate back-polarization energy-drift:           0
            Cation back-polarization energy-drift:              0

        Tail-correction energy-drift:                  0

        Distance constraints energy-drift:                  0
        Angle constraints energy-drift:                     0
        Dihedral constraints energy-drift:                  0
        Inversion-bend constraints energy-drift:                    0
        Out-of-plane distance constraints energy-drift:                    0
        Exclusion constraints energy-drift:                 0

        ===================================================================
        Total energy-drift: 7.107e-10





        Average properties of the system[0]:
        ========================================================================

        Average temperature:
        ====================
            Block[ 0]               -nan [K]
            Block[ 1]               -nan [K]
            Block[ 2]               -nan [K]
            Block[ 3]               -nan [K]
            Block[ 4]               -nan [K]
            ------------------------------------------------------------------------------
            Average                 -nan [K] +/-                nan [K]

        Average Pressure:
        =================
            Block[ 0]            0.00000 [Pa]
            Block[ 1]            0.00000 [Pa]
            Block[ 2]            0.00000 [Pa]
            Block[ 3]            0.00000 [Pa]
            Block[ 4]            0.00000 [Pa]
            ------------------------------------------------------------------------------
            Average              0.00000 [Pa] +/-            0.00000 [Pa]
            Average              0.00000 [bar] +/-            0.00000 [bar]
            Average              0.00000 [atm] +/-            0.00000 [atm]
            Average              0.00000 [Torr] +/-            0.00000 [Torr]

        Average Volume:
        =================
            Block[ 0]        17237.49273 [A^3]
            Block[ 1]        17237.49273 [A^3]
            Block[ 2]        17237.49273 [A^3]
            Block[ 3]        17237.49273 [A^3]
            Block[ 4]        17237.49273 [A^3]
            ------------------------------------------------------------------------------
            Average          17237.49273 [A^3] +/-            0.00000 [A^3]

        Average Box-lengths:
        ====================
            Block[ 0]           25.83200 [A^3]
            Block[ 1]           25.83200 [A^3]
            Block[ 2]           25.83200 [A^3]
            Block[ 3]           25.83200 [A^3]
            Block[ 4]           25.83200 [A^3]
            ------------------------------------------------------------------------------
            Average Box.ax            25.83200 [A^3] +/-            0.00000 [A^3]

            Block[ 0]           25.83200 [A^3]
            Block[ 1]           25.83200 [A^3]
            Block[ 2]           25.83200 [A^3]
            Block[ 3]           25.83200 [A^3]
            Block[ 4]           25.83200 [A^3]
            ------------------------------------------------------------------------------
            Average Box.by            25.83200 [A^3] +/-            0.00000 [A^3]

            Block[ 0]           25.83200 [A^3]
            Block[ 1]           25.83200 [A^3]
            Block[ 2]           25.83200 [A^3]
            Block[ 3]           25.83200 [A^3]
            Block[ 4]           25.83200 [A^3]
            ------------------------------------------------------------------------------
            Average Box.cz            25.83200 [A^3] +/-            0.00000 [A^3]

            Block[ 0]           90.00000 [A^3]
            Block[ 1]           90.00000 [A^3]
            Block[ 2]           90.00000 [A^3]
            Block[ 3]           90.00000 [A^3]
            Block[ 4]           90.00000 [A^3]
            ------------------------------------------------------------------------------
            Average alpha angle            90.00000 [degrees] +/-            0.00000 [degrees]

            Block[ 0]           90.00000 [A^3]
            Block[ 1]           90.00000 [A^3]
            Block[ 2]           90.00000 [A^3]
            Block[ 3]           90.00000 [A^3]
            Block[ 4]           90.00000 [A^3]
            ------------------------------------------------------------------------------
            Average beta angle            90.00000 [degrees] +/-            0.00000 [degrees]

            Block[ 0]           90.00000 [A^3]
            Block[ 1]           90.00000 [A^3]
            Block[ 2]           90.00000 [A^3]
            Block[ 3]           90.00000 [A^3]
            Block[ 4]           90.00000 [A^3]
            ------------------------------------------------------------------------------
            Average gamma angle            90.00000 [degrees] +/-            0.00000 [degrees]


        Average Surface Area:
        =====================
            Block[ 0] 0.000000 [-]
            Block[ 1] 0.000000 [-]
            Block[ 2] 0.000000 [-]
            Block[ 3] 0.000000 [-]
            Block[ 4] 0.000000 [-]
            ------------------------------------------------------------------------------
            Average surface area:   0.000000 +/- 0.000000 [A^2]
                                    0.000000 +/- 0.000000 [m^2/g]
                                    0.000000 +/- 0.000000 [m^2/cm^3]

            Block[ 0] 0.000000 [-]
            Block[ 1] 0.000000 [-]
            Block[ 2] 0.000000 [-]
            Block[ 3] 0.000000 [-]
            Block[ 4] 0.000000 [-]
            ------------------------------------------------------------------------------
            Average surface area:   0.000000 +/- 0.000000 [A^2]
                                    0.000000 +/- 0.000000 [m^2/g]
                                    0.000000 +/- 0.000000 [m^2/cm^3]

            Block[ 0] 0.000000 [-]
            Block[ 1] 0.000000 [-]
            Block[ 2] 0.000000 [-]
            Block[ 3] 0.000000 [-]
            Block[ 4] 0.000000 [-]
            ------------------------------------------------------------------------------
            Average surface area:   0.000000 +/- 0.000000 [A^2]
                                    0.000000 +/- 0.000000 [m^2/g]
                                    0.000000 +/- 0.000000 [m^2/cm^3]


        Average density:
        =================
            Block[ 0]            1.59999 [kg/m^3]
            Block[ 1]            1.57691 [kg/m^3]
            Block[ 2]            1.57498 [kg/m^3]
            Block[ 3]            1.58701 [kg/m^3]
            Block[ 4]            1.57250 [kg/m^3]
            ------------------------------------------------------------------------------
            Average              1.58228 [kg/m^3] +/-            0.01407 [kg/m^3]

            Average density component 0 [CO2]
            -------------------------------------------------------------
                Block[ 0]            0.68601 [kg/m^3]
                Block[ 1]            0.66948 [kg/m^3]
                Block[ 2]            0.67266 [kg/m^3]
                Block[ 3]            0.67965 [kg/m^3]
                Block[ 4]            0.67626 [kg/m^3]
                ------------------------------------------------------------------------------
                Average              0.67681 [kg/m^3] +/-            0.00795 [kg/m^3]
            Average density component 1 [O2]
            -------------------------------------------------------------
                Block[ 0]            0.48072 [kg/m^3]
                Block[ 1]            0.48658 [kg/m^3]
                Block[ 2]            0.48889 [kg/m^3]
                Block[ 3]            0.47949 [kg/m^3]
                Block[ 4]            0.47363 [kg/m^3]
                ------------------------------------------------------------------------------
                Average              0.48186 [kg/m^3] +/-            0.00751 [kg/m^3]
            Average density component 2 [N2]
            -------------------------------------------------------------
                Block[ 0]            0.43326 [kg/m^3]
                Block[ 1]            0.42085 [kg/m^3]
                Block[ 2]            0.41343 [kg/m^3]
                Block[ 3]            0.42787 [kg/m^3]
                Block[ 4]            0.42260 [kg/m^3]
                ------------------------------------------------------------------------------
                Average              0.42360 [kg/m^3] +/-            0.00928 [kg/m^3]

        Average compressibility Z:
        =========================
            Block[ 0]               -nan [-]
            Block[ 1]               -nan [-]
            Block[ 2]               -nan [-]
            Block[ 3]               -nan [-]
            Block[ 4]               -nan [-]
            ------------------------------------------------------------------------------
            Average                 -nan [-] +/-                nan [-]

        Average Heat Capacity (MC-NPT-ensemble): [1/(kB T^2)]*[<H^2>-<H>^2]
        ===================================================================
            Block[ 0] 0.099702 [J/mol/K]
            Block[ 1] 0.092106 [J/mol/K]
            Block[ 2] 0.093896 [J/mol/K]
            Block[ 3] 0.091282 [J/mol/K]
            Block[ 4] 0.090845 [J/mol/K]
            ------------------------------------------------------------------------------
            Average              0.09357 [J/mol/K] +/-            0.00450 [J/mol/K]
            Average              0.02236 [cal/mol/K] +/-            0.00108 [cal/mol/K]

        Enthalpy of adsorption:
        =======================

            Enthalpy of adsorption component 0 [CO2]
            -------------------------------------------------------------
                Block[ 0] -293.74763         [-]
                Block[ 1] -292.30012         [-]
                Block[ 2] -289.55806         [-]
                Block[ 3] -289.16447         [-]
                Block[ 4] -290.41768         [-]
                ------------------------------------------------------------------------------
                Average           -291.03759 +/-           2.405704 [K]
                                    -2.41982 +/-           0.020002 [KJ/MOL]
                Note: Ug should be subtracted to this value
                Note: The heat of adsorption Q=-H

            Enthalpy of adsorption component 1 [O2]
            -------------------------------------------------------------
                Block[ 0] -282.17075         [-]
                Block[ 1] -282.46059         [-]
                Block[ 2] -281.60219         [-]
                Block[ 3] -281.99761         [-]
                Block[ 4] -280.78659         [-]
                ------------------------------------------------------------------------------
                Average           -281.80354 +/-           0.804311 [K]
                                    -2.34305 +/-           0.006687 [KJ/MOL]
                Note: Ug should be subtracted to this value
                Note: The heat of adsorption Q=-H

            Enthalpy of adsorption component 2 [N2]
            -------------------------------------------------------------
                Block[ 0] -282.12659         [-]
                Block[ 1] -282.39212         [-]
                Block[ 2] -280.86164         [-]
                Block[ 3] -282.19966         [-]
                Block[ 4] -281.48537         [-]
                ------------------------------------------------------------------------------
                Average           -281.81308 +/-           0.784102 [K]
                                    -2.34312 +/-           0.006519 [KJ/MOL]
                Note: Ug should be subtracted to this value
                Note: The heat of adsorption Q=-H

            Total enthalpy of adsorption from components and measured mol-fraction
            ----------------------------------------------------------------------
                Block[ 0] -286.07297         [-]
                Block[ 1] -285.73240         [-]
                Block[ 2] -284.04458         [-]
                Block[ 3] -284.48732         [-]
                Block[ 4] -284.29035         [-]
                ------------------------------------------------------------------------------
                Average           -284.92552 +/-           1.134296 [K]
                                    -2.36900 +/-           0.009431 [KJ/MOL]
                Note: Ug should be subtracted to this value
                Note: The heat of adsorption Q=-H


        derivative of the chemical potential with respect to density (constant T,V):
        ============================================================================
            Block[ 0] 11863736.38965     [-]
            Block[ 1] 11929448.77499     [-]
            Block[ 2] 12146227.14830     [-]
            Block[ 3] 11950967.47786     [-]
            Block[ 4] 12092842.29045     [-]
            ------------------------------------------------------------------------------
            Average       11996644.41625 +/-      146754.606159 [-]





        Average energies of the system[0]:
        ========================================================================

        Average Host Bond stretch energy:
        =================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host UreyBradley stretch energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bend angle energy:
        ===============================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bend angle inversion energy:
        =========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Torsion energy:
        ============================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Improper Torsion energy:
        =====================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bond-Bond cross term energy:
        ===============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bend-Bend cross term energy:
        =========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bond-Bend cross term energy:
        ============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bond-Torsion cross term energy:
        ============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host Bend-Torsion cross term energy:
        ============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bond stretch energy:
        ====================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate UreyBradley stretch energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bend angle energy:
        ====================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bend angle inversion energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Torsion energy:
        =================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Improper Torsion energy:
        ==========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bond-Bond cross term energy:
        ====================================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bend-Bend cross term energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bond-Bend cross term energy:
        =================================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bond-Torsion cross term energy:
        =================================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Bend-Torsion cross term energy:
        =================================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Intra Van der Waals energy:
        =============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Intra charge-charge Coulomb energy:
        =======================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Intra charge-bonddipole Coulomb energy:
        =======================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Adsorbate Intra bonddipole-bonddipole Coulomb energy:
        =======================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bond stretch energy:
        =================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation UreyBradley stretch energy:
        ==========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bend angle energy:
        =================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bend angle inversion energy:
        ===========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Torsion energy:
        ==============================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Improper Torsion energy:
        =======================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bond-Bond cross term energy:
        =================================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bend-Bend cross term energy:
        ===========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bond-Bend cross term energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bond-Torsion cross term energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Bend-Torsion cross term energy:
        ==============================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Intra Van der Waals energy:
        ==========================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Intra charge-charge Coulomb energy:
        ====================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Intra charge-bonddipole Coulomb energy:
        ====================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Cation Intra bonddipole-bonddipole Coulomb energy:
        ====================================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Average Host-Host energy:
        =========================
            Block[ 0] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 1] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 2] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 3] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 4] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   0.00000            Van der Waals: 0.000000           Coulomb: 0.00000            [K]
                  +/- 0.00000                       +/- 0.000000                +/- 0.00000            [K]

        Average Adsorbate-Adsorbate energy:
        ===================================
            Block[ 0] -0.88804           Van der Waals: -0.77971           Coulomb: -0.10833           [K]
            Block[ 1] -0.76404           Van der Waals: -0.67154           Coulomb: -0.09251           [K]
            Block[ 2] -0.59039           Van der Waals: -0.53480           Coulomb: -0.05559           [K]
            Block[ 3] -0.57651           Van der Waals: -0.55149           Coulomb: -0.02502           [K]
            Block[ 4] -0.73736           Van der Waals: -0.62484           Coulomb: -0.11252           [K]
            ------------------------------------------------------------------------------
            Average   -0.71127           Van der Waals: -0.632475          Coulomb: -0.07879           [K]
                  +/- 0.16125                       +/- 0.123151                +/- 0.04657            [K]

        Average Cation-Cation energy:
        =============================
            Block[ 0] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 1] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 2] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 3] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 4] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   0.00000            Van der Waals: 0.000000           Coulomb: 0.00000            [K]
                  +/- 0.00000                       +/- 0.000000                +/- 0.00000            [K]

        Average Host-Adsorbate energy:
        ==============================
            Block[ 0] -4.25007           Van der Waals: -4.25007           Coulomb: 0.00000            [K]
            Block[ 1] -4.31372           Van der Waals: -4.31372           Coulomb: 0.00000            [K]
            Block[ 2] -4.00224           Van der Waals: -4.00224           Coulomb: 0.00000            [K]
            Block[ 3] -4.18849           Van der Waals: -4.18849           Coulomb: 0.00000            [K]
            Block[ 4] -3.78324           Van der Waals: -3.78324           Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   -4.10755           Van der Waals: -4.107551          Coulomb: 0.00000            [K]
                  +/- 0.26744                       +/- 0.267445                +/- 0.00000            [K]

        Average Host-Cation energy:
        ===========================
            Block[ 0] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 1] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 2] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 3] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 4] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   0.00000            Van der Waals: 0.000000           Coulomb: 0.00000            [K]
                  +/- 0.00000                       +/- 0.000000                +/- 0.00000            [K]

        Average Adsorbate-Cation energy:
        ================================
            Block[ 0] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 1] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 2] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 3] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            Block[ 4] 0.00000            Van der Waals: 0.00000            Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   0.00000            Van der Waals: 0.000000           Coulomb: 0.00000            [K]
                  +/- 0.00000                       +/- 0.000000                +/- 0.00000            [K]

        Host polarization energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Adsorbate polarization energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Cation polarization energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Host back-polarization energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Adsorbate back-polarization energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Cation back-polarization energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Tail-correction energy:
        =======================
            Block[ 0]           -0.24972 [K]
            Block[ 1]           -0.24764 [K]
            Block[ 2]           -0.24698 [K]
            Block[ 3]           -0.24846 [K]
            Block[ 4]           -0.24732 [K]
            ------------------------------------------------------------------------------
            Average             -0.24803 [K] +/-            0.00136 [K]

        Distance-constraints energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Angle-constraints energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Dihedral-constraints energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Inversion-bend constraints energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Out-of-plane-distance constraints energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Exclusion-constraints energy:
        =======================
            Block[ 0]            0.00000 [K]
            Block[ 1]            0.00000 [K]
            Block[ 2]            0.00000 [K]
            Block[ 3]            0.00000 [K]
            Block[ 4]            0.00000 [K]
            ------------------------------------------------------------------------------
            Average              0.00000 [K] +/-            0.00000 [K]

        Total energy:
        =============
            Block[ 0]           -5.38783 [K]
            Block[ 1]           -5.32541 [K]
            Block[ 2]           -4.83960 [K]
            Block[ 3]           -5.01346 [K]
            Block[ 4]           -4.76792 [K]
            ------------------------------------------------------------------------------
            Average             -5.06684 [K] +/-            0.34768 [K]

        Number of molecules:
        ====================

        Component 0 [CO2]
        -------------------------------------------------------------
            Block[ 0] 0.16185            [-]
            Block[ 1] 0.15795            [-]
            Block[ 2] 0.15870            [-]
            Block[ 3] 0.16035            [-]
            Block[ 4] 0.15955            [-]
            ------------------------------------------------------------------------------
            Average loading absolute                              0.1596800000 +/-       0.0018753930 [-]
            Average loading absolute [molecules/unit cell]        0.1596800000 +/-       0.0018753930 [-]
            Average loading absolute [mol/kg framework]                  1.1876738402 +/-       0.0139488674 [-]
            Average loading absolute [milligram/gram framework]         52.2562237613 +/-       0.6137334279 [-]
            Average loading absolute [cm^3 (STP)/gr framework]          26.6204926509 +/-       0.3126495761 [-]
            Average loading absolute [cm^3 (STP)/cm^3 framework]         0.3447818012 +/-       0.0040493572 [-]

            Block[ 0] 0.00707            [-]
            Block[ 1] 0.00317            [-]
            Block[ 2] 0.00392            [-]
            Block[ 3] 0.00557            [-]
            Block[ 4] 0.00477            [-]
            ------------------------------------------------------------------------------
            Average loading excess                              0.0048987839 +/-       0.0018753930 [-]
            Average loading excess [molecules/unit cell]        0.0048987839 +/-       0.0018753930 [-]
            Average loading excess [mol/kg framework]                    0.0364363571 +/-       0.0139488674 [-]
            Average loading excess [milligram/gram framework]            1.6031559886 +/-       0.6137334279 [-]
            Average loading excess [cm^3 (STP)/gr framework]             0.8166836243 +/-       0.3126495761 [-]
            Average loading excess [cm^3 (STP)/cm^3 framework]           0.0105774771 +/-       0.0040493572 [-]

        Component 1 [O2]
        -------------------------------------------------------------
            Block[ 0] 0.15595            [-]
            Block[ 1] 0.15785            [-]
            Block[ 2] 0.15860            [-]
            Block[ 3] 0.15555            [-]
            Block[ 4] 0.15365            [-]
            ------------------------------------------------------------------------------
            Average loading absolute                              0.1563200000 +/-       0.0024357719 [-]
            Average loading absolute [molecules/unit cell]        0.1563200000 +/-       0.0024357719 [-]
            Average loading absolute [mol/kg framework]                  1.1626827073 +/-       0.0181168750 [-]
            Average loading absolute [milligram/gram framework]         37.2044514148 +/-       0.5797182605 [-]
            Average loading absolute [cm^3 (STP)/gr framework]          26.0603420039 +/-       0.4060711974 [-]
            Average loading absolute [cm^3 (STP)/cm^3 framework]         0.3375268735 +/-       0.0052593301 [-]

            Block[ 0] 0.00117            [-]
            Block[ 1] 0.00307            [-]
            Block[ 2] 0.00382            [-]
            Block[ 3] 0.00077            [-]
            Block[ 4] -0.00113           [-]
            ------------------------------------------------------------------------------
            Average loading excess                              0.0015387839 +/-       0.0024357719 [-]
            Average loading excess [molecules/unit cell]        0.0015387839 +/-       0.0024357719 [-]
            Average loading excess [mol/kg framework]                    0.0114452242 +/-       0.0181168750 [-]
            Average loading excess [milligram/gram framework]            0.3662334396 +/-       0.5797182605 [-]
            Average loading excess [cm^3 (STP)/gr framework]             0.2565329773 +/-       0.4060711974 [-]
            Average loading excess [cm^3 (STP)/cm^3 framework]           0.0033225494 +/-       0.0052593301 [-]

        Component 2 [N2]
        -------------------------------------------------------------
            Block[ 0] 0.16055            [-]
            Block[ 1] 0.15595            [-]
            Block[ 2] 0.15320            [-]
            Block[ 3] 0.15855            [-]
            Block[ 4] 0.15660            [-]
            ------------------------------------------------------------------------------
            Average loading absolute                              0.1569700000 +/-       0.0034394965 [-]
            Average loading absolute [molecules/unit cell]        0.1569700000 +/-       0.0034394965 [-]
            Average loading absolute [mol/kg framework]                  1.1675173015 +/-       0.0255824148 [-]
            Average loading absolute [milligram/gram framework]         32.7062225746 +/-       0.7166524665 [-]
            Average loading absolute [cm^3 (STP)/gr framework]          26.1687044802 +/-       0.5734036258 [-]
            Average loading absolute [cm^3 (STP)/cm^3 framework]         0.3389303566 +/-       0.0074265769 [-]

            Block[ 0] 0.00577            [-]
            Block[ 1] 0.00117            [-]
            Block[ 2] -0.00158           [-]
            Block[ 3] 0.00377            [-]
            Block[ 4] 0.00182            [-]
            ------------------------------------------------------------------------------
            Average loading excess                              0.0021887839 +/-       0.0034394965 [-]
            Average loading excess [molecules/unit cell]        0.0021887839 +/-       0.0034394965 [-]
            Average loading excess [mol/kg framework]                    0.0162798183 +/-       0.0255824148 [-]
            Average loading excess [milligram/gram framework]            0.4560543657 +/-       0.7166524665 [-]
            Average loading excess [cm^3 (STP)/gr framework]             0.3648954537 +/-       0.5734036258 [-]
            Average loading excess [cm^3 (STP)/cm^3 framework]           0.0047260324 +/-       0.0074265769 [-]


        Average Widom Rosenbluth factor:
        ================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Widom Rosenbluth-weight:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Widom Rosenbluth-weight:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Widom Rosenbluth-weight:   0 +/- 0.000000 [-]

        Average Widom chemical potential:
        =================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average chemical potential:   0 +/- 0.000000 [K]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average chemical potential:   0 +/- 0.000000 [K]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average chemical potential:   0 +/- 0.000000 [K]

        Average Widom Ideal-gas contribution:
        =====================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Widom Ideal-gas chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Widom Ideal-gas chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Widom Ideal-gas chemical potential:   0 +/- 0.000000 [-]

        Average Widom excess contribution:
        ==================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Widom excess chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Widom excess chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Widom excess chemical potential:   0 +/- 0.000000 [-]

        Average Gibbs Widom Rosenbluth factor:
        ======================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Gibbs Widom Rosenbluth-weight:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Gibbs Widom Rosenbluth-weight:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Gibbs Widom Rosenbluth-weight:   0 +/- 0.000000 [-]

        Average Gibbs Widom chemical potential:
        =======================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Gibbs chemical potential:   0 +/- 0.000000 [K]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Gibbs chemical potential:   0 +/- 0.000000 [K]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Gibbs chemical potential:   0 +/- 0.000000 [K]

        Average Gibbs Widom Ideal-gas contribution:
        ===========================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Gibbs Ideal-gas chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Gibbs Ideal-gas chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Gibbs Ideal-gas chemical potential:   0 +/- 0.000000 [-]

        Average Gibbs Widom excess contribution:
        ===========================================
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [CO2] Average Gibbs excess chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [O2] Average Gibbs excess chemical potential:   0 +/- 0.000000 [-]
            Block[ 0] 0 [-]
            Block[ 1] 0 [-]
            Block[ 2] 0 [-]
            Block[ 3] 0 [-]
            Block[ 4] 0 [-]
            ------------------------------------------------------------------------------
            [N2] Average Gibbs excess chemical potential:   0 +/- 0.000000 [-]

        Average Henry coefficient:
        ==========================
            Block[ 0] 0 [mol/kg/Pa]
            Block[ 1] 0 [mol/kg/Pa]
            Block[ 2] 0 [mol/kg/Pa]
            Block[ 3] 0 [mol/kg/Pa]
            Block[ 4] 0 [mol/kg/Pa]
            ------------------------------------------------------------------------------
            [CO2] Average Henry coefficient:  0 +/- 0 [mol/kg/Pa]
            Block[ 0] 0 [mol/kg/Pa]
            Block[ 1] 0 [mol/kg/Pa]
            Block[ 2] 0 [mol/kg/Pa]
            Block[ 3] 0 [mol/kg/Pa]
            Block[ 4] 0 [mol/kg/Pa]
            ------------------------------------------------------------------------------
            [O2] Average Henry coefficient:  0 +/- 0 [mol/kg/Pa]
            Block[ 0] 0 [mol/kg/Pa]
            Block[ 1] 0 [mol/kg/Pa]
            Block[ 2] 0 [mol/kg/Pa]
            Block[ 3] 0 [mol/kg/Pa]
            Block[ 4] 0 [mol/kg/Pa]
            ------------------------------------------------------------------------------
            [N2] Average Henry coefficient:  0 +/- 0 [mol/kg/Pa]

        Average adsorption energy <U_gh>_1-<U_h>_0 obtained from Widom-insertion:
        (Note: the total heat of adsorption is dH=<U_gh>_1-<U_h>_0 - <U_g> - RT)
        =========================================================================

        Simulation finished,  1 warnings
        WARNING: INAPPROPRIATE NUMBER OF UNIT CELLS USED


        Tue Sep 26 01:26:50 2023
        Simulation finished on Tuesday, September 26.
        The end time was 01:26 AM.

        """
result = RaspaParser(text)
data = result['CO2']