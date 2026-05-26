#ifndef USB_TASK_H
#define USB_TASK_H

#include "struct_typedef.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define USB_FRAME_DATA_MAX_SIZE      128u
#define USB_FRAME_HEADER_SIZE        4u
#define USB_FRAME_TIMESTAMP_SIZE     4u
#define USB_FRAME_CRC16_SIZE         2u
#define USB_FRAME_MAX_SIZE           \
    (USB_FRAME_HEADER_SIZE + USB_FRAME_TIMESTAMP_SIZE + USB_FRAME_DATA_MAX_SIZE + USB_FRAME_CRC16_SIZE)

#define UPPER_COMMAND_NONE        0u
#define UPPER_COMMAND_START_SCAN  1u
#define UPPER_COMMAND_HOLD        2u
#define UPPER_COMMAND_RESUME_NAV  3u
#define UPPER_COMMAND_START_DUMP  4u

#define LOWER_MODE_IDLE                 0u
#define LOWER_MODE_AUTO_SCAN            1u
#define LOWER_MODE_PICKING              2u
#define LOWER_MODE_PICK_DONE            3u
#define LOWER_MODE_SCAN_DONE_NO_TARGET  4u
#define LOWER_MODE_DUMPING              5u
#define LOWER_MODE_DUMP_DONE            6u
#define LOWER_MODE_ERROR                255u

typedef enum
{
    USB_STEP_HEADER_SOF = 0,
    USB_STEP_LENGTH_ID_CRC8,
    USB_STEP_TIMESTAMP,
    USB_STEP_DATA,
    USB_STEP_CHECKSUM,
} usb_unpack_step_e;

typedef struct
{
    float vx;
    float vy;
    float wz;
} usb_speed_vector_t;

typedef struct
{
    float roll;
    float pitch;
    float yaw;
    float leg_length;
} usb_chassis_t;

typedef struct
{
    float pitch;
    float yaw;
} usb_gimbal_t;

typedef struct
{
    uint8_t fire;
    uint8_t frc_on;
} usb_shoot_t;

typedef struct
{
    bool_t tracking;
} usb_track_t;

typedef struct
{
    usb_speed_vector_t speed_vector;
    usb_chassis_t chassis;
    usb_gimbal_t gimbal;
    usb_shoot_t shoot;
    usb_track_t track;
} usb_data_t;

typedef struct
{
    uint8_t SOF;
    uint8_t LEN;
    uint8_t ID;
    uint8_t CRC8;
} usb_frame_header_t;

typedef struct
{
    usb_frame_header_t header;
    uint32_t timestamp;
    usb_data_t data;
    uint16_t checksum;
} usb_data_frame_t;

typedef struct
{
    uint8_t game_progress;
    uint16_t stage_remain_time;
} usb_game_status_t;

typedef struct
{
    usb_frame_header_t header;
    uint32_t timestamp;
    usb_game_status_t data;
    uint16_t checksum;
} usb_game_status_frame_t;

typedef struct
{
    usb_frame_header_t *p_header;
    uint8_t protocol_packet[USB_FRAME_MAX_SIZE];
    usb_unpack_step_e unpack_step;
    uint16_t data_len;
    uint16_t index;
} usb_unpack_data_t;

typedef enum
{
    GAME_UNSTART = 0,
    GAME_SETUP,
    GAME_MOVING_TO_B,
    GAME_MOVING_TO_C,
    GAME_MOVING_C_TO_TP,
    GAME_MOVING_TO_D,
    GAME_MOVING_D_TO_TP,
    GAME_MOVING_TO_E,
    GAME_MOVING_E_TO_TP,
    GAME_MOVING_TO_F,
    GAME_ADJUST_B,
    GAME_ADJUST_C,
    GAME_ADJUST_D,
    GAME_ADJUST_E,
    GAME_ADJUST_TP,
    GAME_CATCHING,
    GAME_LAYING,
    GAME_FINISHED,
} process_mode_e;

typedef struct __attribute__((packed))
{
    uint8_t head[2];   // 'S','P'
    uint8_t command;
    uint8_t tracking;
    uint8_t class_id;
    float x;
    float y;
    float z;
    usb_speed_vector_t speed_vector;
    uint16_t crc16;
} vision_to_gimbal_packet_t;

typedef struct __attribute__((packed))
{
    uint8_t head[2];   // 'V','S'
    uint8_t robot_mode;
    uint16_t crc16;
} gimbal_to_vision_packet_t;

typedef struct
{
    uint8_t label;
    float distance;
    float dis_x;
    float dis_y;
} pi;

extern pi my_dis;
extern uint8_t receive_flag;
extern uint8_t vision_command_pending;
extern uint8_t vision_last_command;
extern usb_data_t usb_final_data;
extern process_mode_e process_mode;

void vision_start(void);
void vision_uart_watchdog_update(void);
uint8_t vision_send_robot_mode_usb(uint8_t robot_mode);
uint8_t vision_send_current_robot_mode_usb(void);
uint8_t vision_send_mode_usb(uint8_t mode);
uint8_t vision_send_current_mode_usb(void);
void vision_set_robot_mode(uint8_t robot_mode);
uint8_t vision_get_robot_mode(void);
uint8_t vision_consume_command(void);
void vision_usb_rx_bytes(const uint8_t *buf, uint32_t len);
void usb_task(void const *argument);

#ifdef __cplusplus
}
#endif

#endif
