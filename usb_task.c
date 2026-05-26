#include "usb_task.h"

#include "cmsis_os.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void usb_device_init_once(void);
static uint16_t vision_crc16_modbus(const uint8_t *data, uint32_t len);
static uint8_t vision_label_used_in_catching(uint8_t label_id);
static uint8_t vision_label_used_in_laying(uint8_t label_id);
static void vision_apply_packet(const vision_to_gimbal_packet_t *pkt);
static uint8_t vision_unpack_byte(uint8_t byte, vision_to_gimbal_packet_t *out_pkt);

pi my_dis;
usb_data_t usb_final_data;
process_mode_e process_mode = GAME_UNSTART;
uint8_t receive_flag = 0u;
uint8_t vision_command_pending = 0u;
uint8_t vision_last_command = UPPER_COMMAND_NONE;

extern uint8_t step_flag;
extern uint8_t label[8];

static uint8_t vision_rx_buf[sizeof(vision_to_gimbal_packet_t)];
static uint16_t vision_rx_index = 0u;
static uint8_t vision_rx_state = 0u;
static uint32_t vision_last_rx_tick = 0u;
static uint8_t vision_last_label = 0xFFu;
static uint8_t vision_same_label_count = 0u;
static uint8_t vision_last_command_rx = UPPER_COMMAND_NONE;
static uint8_t vision_current_robot_mode = LOWER_MODE_IDLE;
static uint8_t usb_device_inited = 0u;

void usb_task(void const *argument)
{
    uint8_t tx_divider = 0u;

    (void)argument;

    usb_device_init_once();
    vision_start();

    for (;;)
    {
        vision_uart_watchdog_update();

        tx_divider++;
        if (tx_divider >= 5u)
        {
            tx_divider = 0u;
            (void)vision_send_current_robot_mode_usb();
        }

        osDelay(10);
    }
}

void CDC_Receive_FS_UserCallback(uint8_t *buf, uint32_t len)
{
    if (buf == NULL || len == 0u)
    {
        return;
    }

    vision_usb_rx_bytes(buf, len);
}

static void usb_device_init_once(void)
{
    if (usb_device_inited != 0u)
    {
        return;
    }

    MX_USB_DEVICE_Init();
    usb_device_inited = 1u;
}

static uint16_t vision_crc16_modbus(const uint8_t *data, uint32_t len)
{
    uint16_t crc = 0xFFFFu;
    uint32_t i;
    uint8_t j;

    if (data == NULL)
    {
        return crc;
    }

    for (i = 0u; i < len; i++)
    {
        crc ^= data[i];
        for (j = 0u; j < 8u; j++)
        {
            if ((crc & 0x0001u) != 0u)
            {
                crc = (crc >> 1) ^ 0xA001u;
            }
            else
            {
                crc >>= 1;
            }
        }
    }

    return crc;
}

static uint8_t vision_label_used_in_catching(uint8_t label_id)
{
    return (label[0] == label_id ||
            label[1] == label_id ||
            label[2] == label_id ||
            label[3] == label_id);
}

static uint8_t vision_label_used_in_laying(uint8_t label_id)
{
    return (label[4] == label_id ||
            label[5] == label_id ||
            label[6] == label_id ||
            label[7] == label_id);
}

static void vision_apply_command(uint8_t command)
{
    if (command == vision_last_command_rx)
    {
        return;
    }

    vision_last_command_rx = command;
    if (command == UPPER_COMMAND_NONE)
    {
        return;
    }

    vision_last_command = command;
    vision_command_pending = 1u;

    if (command == UPPER_COMMAND_START_SCAN)
    {
        if (vision_current_robot_mode == LOWER_MODE_IDLE ||
            vision_current_robot_mode == LOWER_MODE_PICK_DONE ||
            vision_current_robot_mode == LOWER_MODE_SCAN_DONE_NO_TARGET ||
            vision_current_robot_mode == LOWER_MODE_DUMP_DONE)
        {
            vision_set_robot_mode(LOWER_MODE_AUTO_SCAN);
        }
    }
    else if (command == UPPER_COMMAND_START_DUMP)
    {
        if (vision_current_robot_mode == LOWER_MODE_IDLE ||
            vision_current_robot_mode == LOWER_MODE_PICK_DONE ||
            vision_current_robot_mode == LOWER_MODE_SCAN_DONE_NO_TARGET)
        {
            vision_set_robot_mode(LOWER_MODE_DUMPING);
        }
    }
    else if (command == UPPER_COMMAND_RESUME_NAV)
    {
        vision_set_robot_mode(LOWER_MODE_IDLE);
    }
}

static void vision_apply_packet(const vision_to_gimbal_packet_t *pkt)
{
    if (pkt == NULL)
    {
        return;
    }

    vision_apply_command(pkt->command);

    usb_final_data.speed_vector = pkt->speed_vector;
    usb_final_data.track.tracking = (pkt->tracking != 0u) ? 1u : 0u;

    if (pkt->tracking == 0u)
    {
        receive_flag = 0u;
        vision_same_label_count = 0u;
        return;
    }

    my_dis.label = pkt->class_id;
    my_dis.dis_x = pkt->x;
    my_dis.dis_y = pkt->y;
    my_dis.distance = pkt->z;

    receive_flag = 1u;
    vision_last_rx_tick = HAL_GetTick();

    if (vision_last_label == pkt->class_id)
    {
        if (vision_same_label_count < 255u)
        {
            vision_same_label_count++;
        }
    }
    else
    {
        vision_last_label = pkt->class_id;
        vision_same_label_count = 1u;
    }

    if (vision_same_label_count < 5u)
    {
        return;
    }

    if (process_mode == GAME_CATCHING)
    {
        if (!vision_label_used_in_catching(my_dis.label) && my_dis.label <= 3u && step_flag == 0u)
        {
            step_flag = 1u;
        }
    }
    else if (process_mode == GAME_LAYING)
    {
        if (!vision_label_used_in_laying(my_dis.label) && my_dis.label >= 4u && step_flag == 0u)
        {
            step_flag = 1u;
        }
    }
}

static uint8_t vision_unpack_byte(uint8_t byte, vision_to_gimbal_packet_t *out_pkt)
{
    switch (vision_rx_state)
    {
        case 0u:
            if (byte == 'S')
            {
                vision_rx_buf[0] = byte;
                vision_rx_state = 1u;
                vision_rx_index = 1u;
            }
            break;

        case 1u:
            if (byte == 'P')
            {
                vision_rx_buf[1] = byte;
                vision_rx_index = 2u;
                vision_rx_state = 2u;
            }
            else if (byte == 'S')
            {
                vision_rx_buf[0] = 'S';
                vision_rx_state = 1u;
                vision_rx_index = 1u;
            }
            else
            {
                vision_rx_state = 0u;
                vision_rx_index = 0u;
            }
            break;

        case 2u:
            if (vision_rx_index < sizeof(vision_rx_buf))
            {
                vision_rx_buf[vision_rx_index++] = byte;
            }
            else
            {
                vision_rx_state = 0u;
                vision_rx_index = 0u;
                break;
            }

            if (vision_rx_index >= sizeof(vision_to_gimbal_packet_t))
            {
                vision_to_gimbal_packet_t pkt;
                uint16_t calc_crc;

                memcpy(&pkt, vision_rx_buf, sizeof(pkt));
                calc_crc = vision_crc16_modbus((uint8_t *)&pkt, sizeof(pkt) - 2u);

                if (pkt.head[0] == 'S' && pkt.head[1] == 'P' && calc_crc == pkt.crc16)
                {
                    memcpy(out_pkt, &pkt, sizeof(pkt));
                    vision_rx_state = 0u;
                    vision_rx_index = 0u;
                    return 1u;
                }

                if (byte == 'S')
                {
                    vision_rx_buf[0] = 'S';
                    vision_rx_state = 1u;
                    vision_rx_index = 1u;
                }
                else
                {
                    vision_rx_state = 0u;
                    vision_rx_index = 0u;
                }
            }
            break;

        default:
            vision_rx_state = 0u;
            vision_rx_index = 0u;
            break;
    }

    return 0u;
}

void vision_usb_rx_bytes(const uint8_t *buf, uint32_t len)
{
    vision_to_gimbal_packet_t pkt;
    uint32_t i;

    if (buf == NULL || len == 0u)
    {
        return;
    }

    for (i = 0u; i < len; i++)
    {
        if (vision_unpack_byte(buf[i], &pkt) != 0u)
        {
            vision_apply_packet(&pkt);
        }
    }
}

void vision_start(void)
{
    vision_rx_state = 0u;
    vision_rx_index = 0u;
    receive_flag = 0u;
    vision_command_pending = 0u;
    vision_last_command = UPPER_COMMAND_NONE;
    vision_last_command_rx = UPPER_COMMAND_NONE;
    vision_same_label_count = 0u;
    vision_last_label = 0xFFu;
    vision_last_rx_tick = 0u;
    vision_current_robot_mode = LOWER_MODE_IDLE;
    memset(&my_dis, 0, sizeof(my_dis));
    memset(&usb_final_data, 0, sizeof(usb_final_data));
}

void vision_uart_watchdog_update(void)
{
    if (receive_flag != 0u)
    {
        if ((HAL_GetTick() - vision_last_rx_tick) > 200u)
        {
            receive_flag = 0u;
            vision_same_label_count = 0u;
            usb_final_data.track.tracking = 0u;
        }
    }
}

void vision_set_robot_mode(uint8_t robot_mode)
{
    vision_current_robot_mode = robot_mode;
}

uint8_t vision_get_robot_mode(void)
{
    return vision_current_robot_mode;
}

uint8_t vision_consume_command(void)
{
    uint8_t command = UPPER_COMMAND_NONE;

    if (vision_command_pending != 0u)
    {
        command = vision_last_command;
        vision_command_pending = 0u;
    }

    return command;
}

uint8_t vision_send_robot_mode_usb(uint8_t robot_mode)
{
    gimbal_to_vision_packet_t tx_pkt;

    tx_pkt.head[0] = 'V';
    tx_pkt.head[1] = 'S';
    tx_pkt.robot_mode = robot_mode;
    tx_pkt.crc16 = vision_crc16_modbus((uint8_t *)&tx_pkt, sizeof(tx_pkt) - 2u);

    return (CDC_Transmit_FS((uint8_t *)&tx_pkt, sizeof(tx_pkt)) == USBD_OK) ? 1u : 0u;
}

uint8_t vision_send_current_robot_mode_usb(void)
{
    return vision_send_robot_mode_usb(vision_current_robot_mode);
}

uint8_t vision_send_mode_usb(uint8_t mode)
{
    return vision_send_robot_mode_usb(mode);
}

uint8_t vision_send_current_mode_usb(void)
{
    return vision_send_current_robot_mode_usb();
}
