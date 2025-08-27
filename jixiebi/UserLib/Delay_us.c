#include "main.h"
#include "gpio.h"
#include "tim.h"
void Delay_us(uint32_t time){
	__HAL_TIM_SetAutoreload(&htim1,time);
	__HAL_TIM_SetCounter(&htim1,0);
	HAL_TIM_Base_Start(&htim1);
	while(__HAL_TIM_GetCounter(&htim1)!=time);
	HAL_TIM_Base_Stop(&htim1);
}
