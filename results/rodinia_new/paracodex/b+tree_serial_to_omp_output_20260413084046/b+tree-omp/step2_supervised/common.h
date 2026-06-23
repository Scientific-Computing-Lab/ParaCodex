#ifndef BPTREE_COMMON_H
#define BPTREE_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define DEFAULT_ORDER 508
#define Version "1.0"

typedef struct record {
	int value;
} record;

typedef struct knode {
	int location;
	int indices[DEFAULT_ORDER + 1];
	int keys[DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode;

typedef struct node {
	int *keys;
	void **pointers;
	bool is_leaf;
	int num_keys;
	struct node *parent;
	struct node *next;
} node;

typedef struct list_item_t {
	struct list_item_t *pred;
	struct list_item_t *next;
	void *datum;
} list_item_t;

typedef struct list_t {
	list_item_t *head;
	list_item_t *tail;
	uint32_t length;
	int32_t (*compare)(const void *key, const void *with);
	void (*datum_delete)(void *datum);
} list_t;

typedef list_item_t *list_iterator_t;
typedef list_item_t *list_reverse_iterator_t;

#endif
